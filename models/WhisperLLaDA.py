# ./models/WhisperLLaDA.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import logging
from models.Qformer import BertConfig, BertLMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WhisperLLaDA(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width

        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        ) # learnable query tokens
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, 
                 whisper_model: str,
                 llada_model: str,
                 audio_proj_dim: int = 4096,
                 gen_len: int = 128,
                 mask_id: int = 126336,
                 lora: bool = True,
                 lora_rank: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 second_per_window: float = 0.333333,
                 second_stride: float = 0.333333,
                 task_prompt: str = "Transcribe the audio:"
                 ):
        super().__init__()
        self.second_per_window = second_per_window
        self.second_stride = second_stride
        self.kernel = (1, round(1500 * self.second_per_window / 30.0))
        self.stride = (1, round(1500 * self.second_stride / 30.0))
        self.task_prompt = task_prompt

        # Load whisper and llada
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper = AutoModel.from_pretrained(whisper_model, trust_remote_code=True).encoder.eval().to(self.device)
        for p in self.whisper.parameters():
            p.requires_grad = False # freeze the whisper encoder
        logging.info(f"Whisper model {whisper_model} loaded in")

        self.llada = AutoModelForCausalLM.from_pretrained(llada_model, trust_remote_code=True, torch_dtype=torch.float16).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(llada_model, trust_remote_code=True)
        for p in self.llada.parameters():
            p.requires_grad = False
        logging.info(f"LLaDA model {llada_model} loaded in")

        self.lora = lora
        if self.lora:
            self.loraConfig = LoraConfig(
                                r=lora_rank,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                task_type=TaskType.CAUSAL_LM,
                                inference_mode=False,
                                target_modules=["q_proj", "k_proj", "v_proj"]
                            )
            self.llada = get_peft_model(self.llada, self.loraConfig)
            logging.info(f"LoRA training")

        self.audio_proj_dim = audio_proj_dim
        self.gen_len = gen_len
        self.mask_id = mask_id
        assert self.llada.config.hidden_size == self.audio_proj_dim
                
        self.adapter, self.speech_query_tokens = self.init_speech_Qformer(
                num_query_token=4, speech_width=self.whisper.config.d_model
            )
        self.linear_proj = nn.Linear(in_features=self.adapter.config.hidden_size, out_features=audio_proj_dim)

        for p in self.adapter.parameters():
            p.requires_grad = True

    def _extract_audio_features(self, log_mel):
        with torch.no_grad():
            audio_emb = self.whisper(log_mel, return_dict=True).last_hidden_state

        B, T, C = audio_emb.shape
        speech_embeds_tr = audio_emb.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=self.kernel, dilation=1, padding=0, stride=self.stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, self.kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        audio_emb = speech_embeds_overlap.reshape(-1, self.kernel[1], C)
        speech_atts = torch.ones(audio_emb.size()[:-1], dtype=torch.long, device=audio_emb.device)

        q = self.speech_query_tokens.expand(audio_emb.shape[0], -1, -1)
        output = self.adapter.bert(
            query_embeds=q,
            encoder_hidden_states=audio_emb,
            encoder_attention_mask=speech_atts,
            return_dict=True
        ).last_hidden_state

        feature = self.linear_proj(output)
        feature = feature.view(B, -1, feature.size(2)).contiguous()
        return feature

    def _get_prompt_embeddings(self, batch_size):
        user_input = [{'role': 'user', 'content': self.task_prompt}]
        user_input = self.tokenizer.apply_chat_template(user_input, add_generation_prompt=True, tokenize=False)
        prompt_id = self.tokenizer(user_input)["input_ids"]
        prompt_id = torch.Tensor(prompt_id).to(self.device).unsqueeze(0).to(torch.long)
        
        prompt_id = prompt_id.repeat(batch_size, 1)
        prompt_embedding = self.llada.get_input_embeddings()(prompt_id)
        return prompt_id, prompt_embedding

    def forward(self, samples: dict):
        # print(samples)
        log_mel = samples['spectrogram']
        feature = self._extract_audio_features(log_mel)
        batch_size = feature.shape[0]
        prompt_id, prompt_embedding = self._get_prompt_embeddings(batch_size)
        
        supervision = samples["text"]
        self.tokenizer.pad_token = self.tokenizer.eos_token

        label_tokens = self.tokenizer(
            supervision,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        target = label_tokens.input_ids.to(self.device)

        # Allocate the length of reponse block to the longest true label sentence
        L = target.shape[-1]
        response_block = torch.full((batch_size,L), 0, dtype=torch.long).to(self.device)
        response_block[:,:] = target.clone()

        t = torch.rand(batch_size, device=self.device).unsqueeze(1)
        mask = torch.rand(batch_size, L, device=self.device) < t
        response_block[mask] = self.mask_id
        response_emb = self.llada.get_input_embeddings()(response_block)

        x_emb = torch.cat([prompt_embedding, feature, response_emb], dim=1).to(next(self.llada.parameters()).dtype)
        offset = prompt_embedding.shape[1]+feature.shape[1]

        logits = self.llada(inputs_embeds=x_emb).logits
        masked_logits = logits[..., offset:, :].contiguous()[mask]
        masked_labels = target[..., :].contiguous()[mask]

        loss = F.cross_entropy(masked_logits, masked_labels)
        return loss

    def add_gumbel_noise(self, logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule,
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps # base=0 for mask_num < steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    @torch.inference_mode()
    def generate(self,
                 samples: dict,
                #  correction: bool = False,
                #  mask_ratio: float = 0.3,
                #  confidence_based_masking: bool = True,
                #  num_chunks: int = 4,
                 decode_cfg: dict = None,
                 ):
        log_mel = samples['spectrogram']
        feature = self._extract_audio_features(log_mel)
        batch_size = feature.shape[0]

        prompt_id, prompt_embedding = self._get_prompt_embeddings(batch_size)

        mode = decode_cfg["mode"]
        steps = decode_cfg["steps"]

        x = torch.full((batch_size, prompt_id.shape[1] + feature.shape[1] + self.gen_len), 0, dtype=torch.long).to(self.device)
        x[:, :prompt_id.shape[1]] = prompt_id.clone()
        x[:, prompt_id.shape[1] + feature.shape[1]:] = self.mask_id
        
        gen_block = self.llada.get_input_embeddings()(
            torch.full((batch_size, self.gen_len), self.mask_id, dtype=torch.long).to(self.device)
        )
        x_emb = torch.cat([prompt_embedding, feature, gen_block], dim=1).to(next(self.llada.parameters()).dtype)

        if mode in ["diffusion_deliberation", "semi_ar_deliberation"]:
            mask_ratio = decode_cfg["mask_ratio"]
            confidence_based_masking = (decode_cfg["masking_type"] == "low_confidence")
            num_chunks = decode_cfg["num_chunks"]
            strategy = "diffusion" if mode == "diffusion_deliberation" else "semi_ar"

            out, offset = self.transcribe_correction(
                self.llada, x, x_emb, prompt_id, feature, samples['origin_transcripts'][0], steps=steps,
                mask_ratio=mask_ratio, confidence_based_masking=confidence_based_masking, num_chunks=num_chunks, strategy=strategy
            )
        elif mode == "decoding":
            block_length = decode_cfg["block_length"]
            out, offset = self.transcribe(
                self.llada, x, x_emb, prompt_id, feature, steps=steps, gen_length=self.gen_len, block_length=block_length, cfg_scale=0., remasking='low_confidence'
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out = out[:, offset:]
        result = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return result

    @torch.inference_mode()
    def transcribe(self, model, x, x_emb, prompt, log_mel, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Total sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        with torch.inference_mode():
            prompt_index = (x == mask_id)
            early_stop = False
            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            assert steps % num_blocks == 0
            steps = steps // num_blocks

            for num_block in range(num_blocks):
                block_mask_index = (x[:, prompt.shape[1] + log_mel.shape[1] + num_block * block_length: prompt.shape[1] + log_mel.shape[1] + (num_block + 1) * block_length:] == mask_id)
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
                
                for i in range(steps):
                    mask_index = (x == mask_id)
                    if not mask_index.any():  # If all tokens are unmasked, we do not need to further denoise anything
                        early_stop = True
                        break
                    if cfg_scale > 0.:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(inputs_embeds=x_emb).logits

                    logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                    
                    assert x0.shape == x.shape

                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                    elif remasking == 'random':
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[:, prompt.shape[1] + log_mel.shape[1] + (num_block + 1) * block_length:] = -np.inf

                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                        transfer_index[j, select_index] = True
                    x[transfer_index] = x0[transfer_index]
                    
                    # Mask the later part eos once encounter the first eos token id
                    eos_mask = transfer_index & (x0 == self.tokenizer.eos_token_id)
                    pos = eos_mask.nonzero(as_tuple=True)[1]
                    if pos.numel():
                        first_eos = pos[0].item()
                        x[:,first_eos:] = self.tokenizer.eos_token_id
                        early_stop = True # If we encounter an EOS in current block, we do not need to look into the next block

                    response_embeds = model.get_input_embeddings()(x[:, prompt.shape[1]+log_mel.shape[1]:].to(torch.long))
                    # x_emb = torch.cat([prompt_embeds, log_mel, response_embeds],dim=1).to(next(model.parameters()).dtype)
                    x_emb[:, prompt.shape[1]+log_mel.shape[1]:, :] = response_embeds
                
                if early_stop: break
            return x , prompt.shape[1]+log_mel.shape[1]

    @torch.inference_mode()
    def transcribe_correction(self, model, x, x_emb, prompt, log_mel, supervision, steps=128, temperature=0.,
                            remasking='low_confidence', mask_ratio=0.3, mask_id=126336,
                            confidence_based_masking=True, num_chunks=4, strategy="semi_ar"):
        '''
        Error correction function that uses supervision as initialization and masks tokens for correction.
        
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            log_mel: Log mel spectrogram features.
            supervision: Reference text for correction.
            steps: Sampling steps for correction.
            temperature: Categorical distribution sampling temperature.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_ratio: Ratio of tokens to mask in supervision.
            mask_id: The token id of [MASK] is 126336.
            confidence_based_masking: If True, mask tokens with lowest confidence; if False, mask randomly.
            num_chunks: Number of chunks for semi-autoregressive strategy.
            strategy: 'diffusion' for global masking, or 'semi_ar' for chunk-based masking.
        '''
        with torch.inference_mode():
            supervision_tokens = self.tokenizer(
                supervision, padding=False, truncation=False, 
                return_tensors="pt", add_special_tokens=False
            )
            
            supervision_ids = supervision_tokens.input_ids.to(self.device)
            batch_size = supervision_ids.shape[0]
            gen_length = supervision_ids.shape[1]
            
            response_block = supervision_ids.clone()
            mask_num_per_seq = int(gen_length * mask_ratio)
            
            if strategy == 'diffusion':
                if confidence_based_masking:
                    # First forward pass to get confidence scores for all tokens
                    # Create initial input with padding tokens for response block to avoid model seeing the answer
                    padding_response = torch.full((batch_size, gen_length), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device)
                    padding_response_emb = model.get_input_embeddings()(padding_response)
                    
                    x_initial = torch.cat([
                        model.get_input_embeddings()(prompt.to(torch.long)),
                        log_mel, padding_response_emb
                    ], dim=1).to(next(model.parameters()).dtype)
                    
                    # Get logits for the padded response positions
                    initial_logits = model(inputs_embeds=x_initial).logits
                    response_logits = initial_logits[:, prompt.shape[1]+log_mel.shape[1]:, :]
                    
                    # Calculate confidence scores by comparing predicted tokens with supervision tokens
                    response_probs = F.softmax(response_logits, dim=-1)
                    token_confidences = torch.gather(response_probs, dim=-1, index=supervision_ids.unsqueeze(-1)).squeeze(-1)
                    
                    # Mask tokens with lowest confidence
                    for b in range(batch_size):
                        _, low_confidence_indices = torch.topk(token_confidences[b], k=mask_num_per_seq, largest=False)
                        response_block[b, low_confidence_indices] = mask_id
                        
                    print(f"Using confidence-based masking: masked {mask_num_per_seq} tokens with lowest confidence")
                else:
                    # Random masking
                    for b in range(batch_size):
                        indices_to_mask = torch.randperm(gen_length)[:mask_num_per_seq]
                        response_block[b, indices_to_mask] = mask_id
                        
                    print(f"Using random masking: masked {mask_num_per_seq} random tokens")
                
                x = torch.full((batch_size, prompt.shape[1] + log_mel.shape[1] + gen_length), 0, dtype=torch.long).to(self.device)
                x[:, :prompt.shape[1]] = prompt.clone()
                x[:, prompt.shape[1]+log_mel.shape[1]:] = response_block
                
                response_emb = model.get_input_embeddings()(response_block)
                x_emb = torch.cat([
                    model.get_input_embeddings()(prompt.to(torch.long)),
                    log_mel, response_emb
                ], dim=1).to(next(model.parameters()).dtype)
                
                mask_index = (x == mask_id)
                steps = min(steps, mask_index.sum().item())
                num_transfer_tokens = self.get_num_transfer_tokens(mask_index, steps)
                
                for i in range(steps):
                    mask_index = (x == mask_id)
                    if not mask_index.any():
                        break
                    
                    logits = model(inputs_embeds=x_emb).logits
                    logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    
                    if remasking == 'low_confidence':
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    elif remasking == 'random':
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        if num_transfer_tokens[j, i] > 0: # this ensures the correction steps equals to mask numbers, even if steps is set to 128
                            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                            transfer_index[j, select_index] = True
                    
                    x[transfer_index] = x0[transfer_index]
                    
                    response_embeds = model.get_input_embeddings()(x[:, prompt.shape[1]+log_mel.shape[1]:].to(torch.long))
                    x_emb[:, prompt.shape[1]+log_mel.shape[1]:, :] = response_embeds
                
                return x, prompt.shape[1]+log_mel.shape[1]

            elif strategy == 'semi_ar':
                chunk_size = gen_length // num_chunks
                chunks = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    if i == num_chunks - 1:
                        end_idx = gen_length
                    else:
                        end_idx = (i + 1) * chunk_size
                    chunks.append((start_idx, end_idx))
                
                print(f"Using progressive {num_chunks}-chunk masking strategy")
                
                for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
                    print(f"Processing chunk {chunk_idx + 1}/{num_chunks}: tokens {start_idx} to {end_idx}")
                    
                    response_block = supervision_ids.clone()
                    for b in range(batch_size):
                        response_block[b, start_idx:end_idx] = mask_id
                    
                    x = torch.full((batch_size, prompt.shape[1] + log_mel.shape[1] + gen_length), 0, dtype=torch.long).to(self.device)
                    x[:, :prompt.shape[1]] = prompt.clone()
                    x[:, prompt.shape[1]+log_mel.shape[1]:] = response_block
                    
                    response_emb = model.get_input_embeddings()(response_block)
                    x_emb = torch.cat([
                        model.get_input_embeddings()(prompt.to(torch.long)),
                        log_mel, response_emb
                    ], dim=1).to(next(model.parameters()).dtype)
                    
                    mask_index = (x == mask_id)
                    chunk_steps = steps // num_chunks
                    num_transfer_tokens = self.get_num_transfer_tokens(mask_index, chunk_steps)
                    
                    for i in range(chunk_steps):
                        mask_index = (x == mask_id)
                        if not mask_index.any():
                            break
                        
                        logits = model(inputs_embeds=x_emb).logits
                        logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                        x0 = torch.argmax(logits_with_noise, dim=-1)
                        
                        if remasking == 'low_confidence':
                            p = F.softmax(logits, dim=-1)
                            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                        elif remasking == 'random':
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                        else:
                            raise NotImplementedError(remasking)
                        
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            if num_transfer_tokens[j, i] > 0:
                                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                                transfer_index[j, select_index] = True
                        
                        x[transfer_index] = x0[transfer_index]
                        
                        response_embeds = model.get_input_embeddings()(x[:, prompt.shape[1]+log_mel.shape[1]:].to(torch.long))
                        x_emb[:, prompt.shape[1]+log_mel.shape[1]:, :] = response_embeds
                    
                    supervision_ids[:, start_idx:end_idx] = x[:, prompt.shape[1]+log_mel.shape[1]+start_idx:prompt.shape[1]+log_mel.shape[1]+end_idx]
                    print(f"Completed chunk {chunk_idx + 1}/{num_chunks} correction")
                
                response_block = supervision_ids.clone()
                x = torch.full((batch_size, prompt.shape[1] + log_mel.shape[1] + gen_length), 0, dtype=torch.long).to(self.device)
                x[:, :prompt.shape[1]] = prompt.clone()
                x[:, prompt.shape[1]+log_mel.shape[1]:] = response_block
                
                return x, prompt.shape[1]+log_mel.shape[1]
                
            else:
                raise NotImplementedError(f"Strategy {strategy} not implemented.")
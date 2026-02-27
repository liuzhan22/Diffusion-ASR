# ./file_inference.py

import torch
import json
import soundfile as sf
import logging
import sys
import argparse
import yaml
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from models.WhisperLLaDA import WhisperLLaDA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

parser = argparse.ArgumentParser(description="Run Whisper-LLaDA inference")
parser.add_argument("--config-path", type=str, required=True, help="Path to the decode config file.")
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
    
model_cfg = config['model']
decode_cfg = config['decode']
data_cfg = config['data']

mode = decode_cfg['mode']
is_deliberation = mode in ["diffusion_deliberation", "semi_ar_deliberation"]

logging.info(f"Decoding Mode: {mode}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperLLaDA(
    whisper_model=model_cfg['whisper_path'],
    llada_model=model_cfg['llada_path'],
    gen_len=model_cfg['gen_len'],
    lora=model_cfg['lora'],
    lora_rank=model_cfg['lora_rank'],
    lora_alpha=model_cfg['lora_alpha'],
    lora_dropout=model_cfg['lora_dropout'],
).to(device).eval()

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_cfg['whisper_path'])

if model_cfg['ckpt']:
    ckpt = torch.load(model_cfg['ckpt'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    logging.info(f"Loaded ckpt from {model_cfg['ckpt']}")

with open(data_cfg['test_data'], "r") as f:
    data = json.load(f)

result = []
total_duration = 0.0

for i, item in enumerate(tqdm(data["annotation"], desc="Transcribing")):
    audio, sr = sf.read(item['path'])
    audio = audio[:sr*30]
    duration = audio.shape[-1] / sr
    total_duration += duration

    log_mel = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].to(device)

    samples = {
        'spectrogram': log_mel,
        'text': [item['text']],
        'duration': [duration]
    }

    if is_deliberation:
        samples['origin_transcripts'] = [item['origin_transcripts']]

    predictions = model.generate(samples=samples, decode_cfg=decode_cfg)

    res_item = {
        "path": item['path'],
        "text": item['text'],
        "llada_prediction": predictions[0]
    }
    
    if is_deliberation:
        res_item["origin_transcripts"] = item['origin_transcripts']

    result.append(res_item)

    if i % 10 == 0:
        logging.info(f"[Time: {total_duration:.2f}s] Pred: {predictions[0]}")

with open(data_cfg['output_path'], "w") as f:
    json.dump(result, f, indent=4)

logging.info(f"Done. Saved to {data_cfg['output_path']}")
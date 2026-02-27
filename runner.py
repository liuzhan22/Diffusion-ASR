# ./runner.py

import os
import json
import time
import datetime
from pathlib import Path
import logging
import kaldialign
import wandb
import re
from whisper_normalizer.english import EnglishTextNormalizer
from typing import Dict,  List, Tuple
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from utils.logger import MetricLogger, SmoothedValue
from utils.utils import get_dataloader, prepare_sample
from utils.optims import get_optimizer, LinearWarmupCosineLRScheduler

class Runner:
    def __init__(self, cfg, model, datasets, job_id, ckpt=None):
        self.config = cfg

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_distributed = torch.distributed.is_initialized()
        self.start_epoch = 0
        if ckpt is not None:
            self.start_epoch = ckpt['epoch']
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.cuda_enabled = (self.device.type == "cuda")

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")

        # model
        self._model = model
        self._model.to(self.device)

        # Load model weights from checkpoint if available
        if ckpt is not None and 'model' in ckpt:
            self._model.load_state_dict(ckpt['model'], strict=False)
            logging.info(f"Loaded model state from checkpoint")

        if self.use_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(
                self._model, device_ids=[local_rank], find_unused_parameters=True,
            )
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = get_dataloader(datasets["train"], self.config.config.run, is_train=True, use_distributed=self.use_distributed)
        self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)
        self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        if self.scaler is not None and ckpt is not None:
            if 'scaler' in ckpt and ckpt['scaler'] is not None:
                self.scaler.load_state_dict(ckpt['scaler'])
                logging.info("Loaded scaler state from checkpoint")
            else:
                logging.warning("No scaler state found in checkpoint, starting with fresh scaler")

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)

        if ckpt is not None and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p is None:
                        continue
                    state = self.optimizer.state.get(p, None)
                    if not state:
                        continue
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            want_dtype = p.dtype
                            if k == 'step' and v.device.type == 'cpu' and v.dtype in (torch.float32, torch.float64, torch.int64):
                                continue
                            state[k] = v.to(device=p.device, dtype=want_dtype)

        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        if ckpt is not None and 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

        if is_main_process():
            wandb.init(
                project="LLADA_ASR",
                name=f"train_LibriSpeech960_{job_id}",
                config=self.config.to_dict(),
                dir=str(self.output_dir),
            )

        self.log_config()

        # Normalizer
        self.english = EnglishTextNormalizer()

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            self.scheduler.step(cur_epoch=epoch, cur_step=i)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                loss = self.model(samples)

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            if is_main_process():
                wandb.log({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split):
        model = self.unwrap_dist_model(self.model)
        model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Eval: data epoch: [{}]".format(epoch)

        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                loss = model(samples)

            metric_logger.update(loss=loss.item())

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        
        ret = {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
        }

        if is_main_process():
            wandb.log({
                "epoch": epoch if isinstance(epoch, int) else "-1",
                f"eval_{split}_loss": float(ret["loss"]),
            })

        return ret

    def train(self):
        start_time = time.time()
        best_agg_metric = float("inf")
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid")
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = float(valid_log["loss"])
                    if agg_metrics < best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch
                        self.save_checkpoint(cur_epoch, is_best=True)
                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
        wandb.finish()

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def WER(self, pred, refs):
        ERR = "*"
        num_corr = 0
        subs: Dict[Tuple[str, str], int] = defaultdict(int)
        ins: Dict[str, int] = defaultdict(int)
        dels: Dict[str, int] = defaultdict(int)
        words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for ref, hyp in zip(refs, pred):
            ref = ref.split()
            hyp = hyp.split()
            ali = kaldialign.align(ref, hyp, ERR, sclite_mode=False)
            for ref_word, hyp_word in ali:
                if ref_word == ERR:
                    ins[hyp_word] += 1
                    words[hyp_word][3] += 1
                elif hyp_word == ERR:
                    dels[ref_word] += 1
                    words[ref_word][4] += 1
                elif hyp_word != ref_word:
                    subs[(ref_word, hyp_word)] += 1
                    words[ref_word][1] += 1
                    words[hyp_word][2] += 1
                else:
                    words[ref_word][0] += 1
                    num_corr += 1
        ref_len = sum([len(r.split()) for r in refs])
        sub_errs = sum(subs.values())
        ins_errs = sum(ins.values())
        del_errs = sum(dels.values())
        tot_errs = sub_errs + ins_errs + del_errs
        tot_err_rate = round(100.0 * tot_errs / ref_len, 2)
        print(f"Insertion error: {ins_errs}, deletion error: {del_errs}, substitution error: {sub_errs}, and WER is {((ins_errs+del_errs+sub_errs)*100/ref_len):.2f}%")

        return ins_errs, del_errs, sub_errs, tot_err_rate
                
    def remove_sp(self,text):
        PUNCS = '!,.?;:'
        gt = re.sub(r"<\|.*?\|>", " ", text)
        gt = re.sub(rf"\s+", r" ", gt) 
        gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
        gt = gt.lstrip(" ")
        return gt
    
    def normalize(self, text, prediction):
        text = self.remove_sp(text)
        prediction = self.remove_sp(prediction)
        text = self.english(text)
        prediction = self.english(prediction)

        return text, prediction

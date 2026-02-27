# ./train.py

import argparse
import torch
from utils.utils import *
from utils.config import Config
from utils.dist_utils import init_distributed_mode

from dataset import SALMONNDataset
from runner import Runner
from models.WhisperLLaDA import WhisperLLaDA

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load config
    cfg = Config(parse_args())
    model_config = cfg.config.model
    run_config = cfg.config.run
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode()
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    cfg.pretty_print()

    ckpt_path = model_config.ckpt
    ckpt = torch.load(ckpt_path, map_location=device) if ckpt_path else None

    # build model
    model = WhisperLLaDA(
        whisper_model=model_config.whisper_path,
        llada_model=model_config.llada_path,
        gen_len=model_config.gen_len,
        second_per_window=model_config.second_per_window,
        second_stride=model_config.second_stride,
        task_prompt=model_config.task_prompt,
        lora=model_config.lora,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
    ).to(device)
    
    if ckpt is not None and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        logging.info(f"Load state from checkpoint: {ckpt_path}")
    else:
        logging.info("No checkpoint loaded. Initialize model from scratch.")

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.train_ann_path, model_config.whisper_path),
        "valid": SALMONNDataset(data_config.valid_ann_path, model_config.whisper_path),
        "test": SALMONNDataset(data_config.test_ann_path, model_config.whisper_path),
    }

    # build runner
    runner = Runner(cfg, model, datasets, job_id, ckpt)

    # train
    runner.train()

if __name__ == "__main__":
    main()
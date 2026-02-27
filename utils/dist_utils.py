# ./utils/dist_utils.py

import os
import datetime
import functools
import logging

import torch
import torch.distributed as dist

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        # init_process_group automatically reads from environment variables when using torchrun
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(days=365),
            device_id=torch.device(f"cuda:{local_rank}")
        )
        dist.barrier()
        setup_for_distributed(is_main_process())
    else:
        logging.info("Not using distributed mode")

def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper
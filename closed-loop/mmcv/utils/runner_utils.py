# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import sys
import time
import warnings
from getpass import getuser
from socket import gethostname

import numpy as np
from mmcv.utils import is_str
import functools
import os
import subprocess
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def get_host_info():
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but '
                        f'got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce parameters.

    Args:
        params (list[torch.Parameters]): List of parameters or buffers of a
            model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[torch.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)
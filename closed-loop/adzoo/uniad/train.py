import argparse
import torch
import copy
import os
import time
import warnings
from os import path as osp
from mmcv import __version__ as mmcv_version
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import collect_env, get_root_logger, mkdir_or_exist, set_random_seed, get_dist_info, init_dist, \
                    Config, DictAction, TORCH_VERSION, digit_version
from mmcv.datasets.builder import build_dataloader
from mmcv.optims import build_optimizer
from torch.nn.parallel import DataParallel, DistributedDataParallel
from mmcv.core.evaluation.eval_hooks import CustomDistEvalHook
from mmcv.core import EvalHook
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner)
from adzoo.uniad.test_utils import custom_multi_gpu_test

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    elif args.launcher == 'pytorch':
        torch.backends.cudnn.benchmark = True
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # Create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # meta info
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # seed
    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=args.deterministic)

    # logger
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=cfg.model.type)
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')

    # Dataset
    datasets = [build_dataset(cfg.data.train)]

    # Save meta info
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmcv_version=mmcv_version, config=cfg.pretty_text, CLASSES=datasets[0].CLASSES, \
                                          PALETTE=datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None) # # for segmentors
    
    # Dataloader
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [build_dataloader(ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        # cfg.gpus will be ignored if distributed
                        len(cfg.gpu_ids),
                        dist=distributed,
                        seed=cfg.seed,
                        shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
                        nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
                        ) for ds in datasets
                        ]

    # Model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.CLASSES = datasets[0].CLASSES  # add an attribute for visualization convenience
    logger.info(f'Model:\n{model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=find_unused_parameters
                                        )
    else:
        model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # Optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer_config = OptimizerHook(**cfg.optimizer_config)

    # Runner
    runner = build_runner(cfg.runner, default_args=dict(model=model,
                                                        optimizer=optimizer,
                                                        work_dir=cfg.work_dir,
                                                        logger=logger,
                                                        meta=meta))
    runner.timestamp = timestamp
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    
    # Register eval hooks for interval eval
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        assert False
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(
            cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
    )
    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
    eval_hook = CustomDistEvalHook if distributed else EvalHook
    runner.register_hook(eval_hook(val_dataloader, test_fn=custom_multi_gpu_test, **eval_cfg))

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

if __name__ == '__main__':
    main()

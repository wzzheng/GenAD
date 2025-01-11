# Copyright (c) OpenMMLab. All rights reserved.
import copy
from mmcv.utils import Registry

RUNNERS = Registry('runner')

def build_runner(cfg, default_args=None):
    runner_cfg = copy.deepcopy(cfg)
    runner = RUNNERS.build(runner_cfg, default_args=default_args)
    return runner
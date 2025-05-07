#!/usr/bin/env bash
set -x

# 环境变量
# export TORCH_HOME='/mnt/nas/algorithm/chenming.zhang/.cache/torch'
# export HF_HOME='/mnt/nas/algorithm/chenming.zhang/.cache/huggingface'

# 激活python环境
# source /mnt/nas/algorithm/hao.hong/packages/miniconda3/bin/activate b2d_zoo

# 获取master地址
if [ $MASTER_ADDR == "localhost" ]; then
    master_name=$HOSTNAME
else
    master_name=$MASTER_ADDR
fi
master_ip=$(getent hosts "$master_name" | awk '{ print $1 }' | head -n1)
master_port=$MASTER_PORT

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$master_ip:$master_port \
   $(dirname "$0")/train.py ./adzoo/genad/configs/VAD/GenAD_config_b2d.py --launcher pytorch --deterministic

#python -m torch.distributed.launch --nnodes=2 --nproc_per_node=8 --master_addr=$master_ip --master_port=$master_port --node_rank=$RANK \
#    $(dirname "$0")/train.py ./adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py --launcher pytorch --deterministic
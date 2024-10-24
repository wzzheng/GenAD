#!/bin/bash
set -x
CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

if ! command -v nslookup &> /dev/null; then
    apt update
    apt install dnsutils -y
fi

output=$(nslookup $MY_APP_NAME)
addresses=$(echo "$output" | awk '/^Name:/ { name=$2; next } name && /^Address:/ { print $2 }')
sorted_address_list=($(printf '%s\n' "${addresses[@]}" | sort))

i=0
IFS=' ' read -ra addresses <<< "${sorted_address_list[@]}"

for address in "${addresses[@]}"; do
    POD_IPs[$i]=$address
    i=$((i+1))
done

length=${#POD_IPs[@]}

local_ip=$(hostname -I | grep -oP '\d+\.\d+\.\d+\.\d+')
echo "local ip is $local_ip"
echo "master ip is ${POD_IPs[0]}"

if [ "$local_ip" == ${POD_IPs[0]} ]; then
   #python -m torch.distributed.run --nproc_per_node=8 --master_port=2333 tools/train.py projects/configs/VAD/VAD_tiny_e2e.py --launcher pytorch --deterministic --work-dir ./outputs/VAD_tiny_e2e_v1_ar_test
    source /remote-home/share/miniconda3/bin/activate && conda activate vad && python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=2 --node_rank=0 --master_addr=${POD_IPs[0]} --master_port=$PORT $(dirname "$0")/train.py projects/configs/VAD/VAD_tiny_e2e.py --launcher pytorch ${@:3} --deterministic --work-dir ./outputs/VAD_tiny_e2e_v1_ar_test

fi

if [ "$local_ip" == ${POD_IPs[1]} ]; then
   source /remote-home/share/miniconda3/bin/activate && conda activate vad && python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=2 --node_rank=0 --master_addr=${POD_IPs[0]} --master_port=$PORT $(dirname "$0")/train.py projects/configs/VAD/VAD_tiny_e2e.py --launcher pytorch ${@:3} --deterministic --work-dir ./outputs/VAD_tiny_e2e_v1_ar_test
#    command="python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes 2 --node_rank 1 --master_addr=${POD_IPs[0]} --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic"
fi

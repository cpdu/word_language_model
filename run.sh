#!/bin/bash

module load pytorch
module load python
nvidia-smi
hostname
echo $CUDA_VISIBLE_DEVICES

# parallelization parameters
is_distributed=True
world_size=2
rank=0
shared_file=file://$HOME/shared_file

# training parameters
epoch=1
batch_size=20
bptt=35
nhid=400
data=$HOME/dataset/wikitext-2
model=LSTM
emsize=200
nlayers=2
lr=20
clip=0.25
dropout=0.2
seed=1111
log_interval=200
save=model.pt

# training on single GPU
if [ "$is_distributed" != "True" ]; then
    python3 -u main.py --data ${data} --model ${model} --emsize ${emsize} --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --cuda
    exit 0
fi

# distributed training on multiple GPUs and machines
devices=$CUDA_VISIBLE_DEVICES
OLD_IFS="$IFS"
IFS=","
device_list=($devices)
IFS="$OLD_IFS"
gpu_num=${#device_list[@]}

i=0
while [ "${rank}" != "$(($world_size-1))" -a "${i}" != "$((gpu_num-1))" ]
do
    python3 -u main.py --data ${data} --model ${model} --emsize ${emsize} --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --device ${device_list[$i]} --rank ${rank} --world_size ${world_size} --shared_file ${shared_file} --distributed --cuda &
    rank=$(($rank+1))
    i=$(($i+1))
done
python3 -u main.py --data ${data} --model ${model} --emsize ${emsize} --nlayers ${nlayers} --lr ${lr} --clip ${clip} --dropout ${dropout} --seed ${seed} --log-interval ${log_interval} --save ${save} --nhid ${nhid} --epoch ${epoch} --batch_size ${batch_size} --bptt ${bptt} --device ${device_list[$i]} --rank ${rank} --world_size ${world_size} --shared_file ${shared_file} --distributed --cuda

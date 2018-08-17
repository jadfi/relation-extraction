#!/bin/bash

file="model_nytpcnn_lstmdenseatt"
if [ $# -gt 0 ]; then
    file=$1
fi

if [ ! -d ./model/$file ]; then
    mkdir -p ./model/$file
    mkdir -p ./log/$file
    mkdir -p ./log/$file/train
    mkdir -p ./log/$file/test
    mkdir -p ./stats/$file
fi

rm -r ./model/$file/*
rm -r ./log/$file/train/*
rm -r ./log/$file/test/*
rm -r ./stats/$file/*eval_stats*

for((i=0;i<=0;i++))
do

CUDA_VISIBLE_DEVICES=0 python3 bag_runner.py --name $file --epoch 200 \
    --lrate 0.001 \
    --model_dir ./model/$file --log ./log/$file --eval_dir ./stats/$file \
    --bag_num 50 \
    --vocab_size 80000 \
    --L 145 \
    --entity_dim 5 \
    --enc_dim 230 \
    --cat_n 53 \
    --cell_type pcnn \
    --lrate_decay 0 \
    --report_rate 0.2 \
    --seed 57 \
    --clip_grad 10 \
    --gpu_usage 0.9 \
    --dropout 0.5 \
    --dataset nyt  \
    --att_loss_weight ${i} \
    --softmax_loss 
    #--softmax_loss_size 10
    #--include-NA-loss \
    #--max_eval_rel 3000 \
    
done

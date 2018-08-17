#!/bin/bash

file="model_mypcnn_poollstm_denseattall"
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
#rm -r ./stats/$file/*

for((i=0;i<=0;i++))
do
CUDA_VISIBLE_DEVICES=0 python3 bag_runner.py  --name $file --epoch 10 \
    --lrate 0.001 \
    --embed ../data/vector_np_200d.pkl \
    --model_dir ./model/$file --log ./log/$file --eval_dir ./stats/$file \
    --bag_num 50 \
    --vocab_size 60000 \
    --L 120 \
    --entity_dim 5 \
    --enc_dim 250 \
    --cat_n 5 \
    --cell_type pcnn \
    --lrate_decay 0 \
    --report_rate 0.2 \
    --seed 57 \
    --test_split 1000 \
    --clip_grad 10 \
    --gpu_usage 0.9 \
    --dropout 0.5 \
    --att_loss_weight ${i} \
    --softmax_loss
    # --adv_eps 0.15 \
    # --adv_type sent \
    #--softmax_loss \
    #--adv_eps 0.05 \
    #--softmax_loss \
    #--softmax_loss_size 5 \
    #--tune_embed \
    #--adv_eps 0.05 \
done
echo "done"
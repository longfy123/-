#!/bin/bash

cd /root/MoELLM/diffusion

rm -rf output/model/*

python train.py \
    --data shanghai \
    --data_type original \
    --T_h 12 \
    --T_p 1 \
    --lr 0.0005 \
    --batch_size 8 \
    --N 200 \
    --sample_steps 150 \
    --n_samples 8 > /dev/null 2>&1

MODEL_FILE=$(ls output/model/*.dm4stg 2>/dev/null | head -1)
if [ -n "$MODEL_FILE" ]; then
    mv "$MODEL_FILE" /root/MoELLM/data/shanghai/residual.pt
    echo "The model has been saved : /root/MoELLM/data/shanghai/residual.pt"
else
    echo "error"
fi

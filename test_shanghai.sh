#!/bin/bash

API_KEY="your key"
STATION_IDS=$(seq -s ',' 0 1041)

python main.py \
    --api_key $API_KEY \
    --model_name "gpt-4o-mini" \
    --dataset shanghai \
    --station_ids $STATION_IDS \
    --seed 42 \
    --enable_cuda True \
    --max_workers 20 \
    --timeout 30


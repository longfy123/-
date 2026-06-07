#!/bin/bash

API_KEY="your key"
STATION_IDS=$(seq -s ',' 0 1041)
OUTPUT_DIR="llm_final_results_all"

python main.py \
    --api_key $API_KEY \
    --dataset shanghai \
    --station_ids $STATION_IDS \
    --seed 42 \
    --enable_cuda True \
    --output_dir $OUTPUT_DIR \
    --max_workers 20


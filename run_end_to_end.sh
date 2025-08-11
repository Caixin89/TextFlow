#!/bin/bash


source .env
python src/vqa.py --dataset $DATASET --model_name $END_TO_END_MODEL
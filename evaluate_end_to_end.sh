#!/bin/bash

END_TO_END_MODEL_WITHOUT_PROVIDER=${END_TO_END_MODEL##*/}

source .env
python src/evaluation.py --model_name $JUDGE_MODEL --data_path output/${DATASET}/vqa/${END_TO_END_MODEL_WITHOUT_PROVIDER}.json
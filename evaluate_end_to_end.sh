#!/bin/bash


source .env
python src/evaluation.py --model_name $END_TO_END_MODEL --data_path output/${DATASET}/vqa/${END_TO_END_MODEL}.json
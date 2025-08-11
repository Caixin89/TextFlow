#!/bin/bash


source .env
python src/textualizer.py --dataset $DATASET --textualizer $TEXTUALIZER_MODEL --output_type $FLOWCHART_CODE_FORMAT
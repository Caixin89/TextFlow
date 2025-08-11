#!/bin/bash


source .env

# Check --tool_use is passed
tool_use_args=""
for arg in "$@"; do
  if [[ "$arg" == "--tool_use" ]]; then
    tool_use_args="--tool_use"
    break # Exit loop once found
  fi
done

python src/reasoner.py --dataset $DATASET --reasoner $REASONER_MODEL --textualizer $TEXTUALIZER_MODEL --input_type $FLOWCHART_CODE_FORMAT $tool_use_args
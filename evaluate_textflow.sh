#!/bin/bash


source .env

# Check --tool_use is passed
for arg in "$@"; do
  if [[ "$arg" == "--tool_use" ]]; then
    found_tool_use=true
    break # Exit loop once found
  fi
done

REASONER_MODEL_WITHOUT_PROVIDER=${REASONER_MODEL##*/}
TEXTUALIZER_MODEL_WITHOUT_PROVIDER=${TEXTUALIZER_MODEL##*/}


if [[ -z "$found_verbose" ]]; then
  python src/evaluation.py --model_name $JUDGE_MODEL --data_path output/${DATASET}/textflow/${FLOWCHART_CODE_FORMAT}_reasoner_${REASONER_MODEL_WITHOUT_PROVIDER}_textualizer_${TEXTUALIZER_MODEL_WITHOUT_PROVIDER}.json
else
  python src/evaluation.py --model_name $JUDGE_MODEL --data_path output/${DATASET}/textflow/${FLOWCHART_CODE_FORMAT}_reasoner_tool_use_${REASONER_MODEL_WITHOUT_PROVIDER}_textualizer_${TEXTUALIZER_MODEL_WITHOUT_PROVIDER}.json
fi
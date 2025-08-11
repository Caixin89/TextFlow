#!/bin/bash


source .env

# Check --tool_use is passed
for arg in "$@"; do
  if [[ "$arg" == "--tool_use" ]]; then
    found_tool_use=true
    break # Exit loop once found
  fi
done


if [[ -z "$found_verbose" ]]; then
  python evaluation.py --model_name $REASONER_MODEL --data_path output/${DATASET}/textflow/${FLOWCHART_CODE_FORMAT}_reasoner_${REASONER_MODEL}_textualizer_${TEXTUALIZER_MODEL}.json
else
  python evaluation.py --model_name $REASONER_MODEL --data_path output/${DATASET}/textflow/${FLOWCHART_CODE_FORMAT}_reasoner_tool_use_${REASONER_MODEL}_textualizer_${TEXTUALIZER_MODEL}.json
fi
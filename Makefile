# Device selection: 'gpu' or 'cpu' (override via `make DEVICE=cpu ...`)
DEVICE ?= gpu

ifeq ($(DEVICE),cpu)
COMPOSE_CMD := docker compose run --rm --service-ports flowchart_vqa_cpu bash -lc
else
COMPOSE_CMD := docker compose run --rm --service-ports flowchart_vqa bash -lc
endif

.PHONY: eval_textflow eval_textflow_tool_use eval_end_to_end run_reasoner run_reasoner_tool_use run_textualizer run_end_to_end build

build:
	docker compose build

eval_textflow:
	$(COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/textflow/$${FLOWCHART_CODE_FORMAT}_reasoner_$${REASONER_MODEL##*/}_textualizer_$${TEXTUALIZER_MODEL##*/}.json"'

eval_textflow_tool_use:
	$(COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/textflow/$${FLOWCHART_CODE_FORMAT}_reasoner_tool_use_$${REASONER_MODEL##*/}_textualizer_$${TEXTUALIZER_MODEL##*/}.json"'

eval_end_to_end:
	$(COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/vqa/$${END_TO_END_MODEL##*/}.json"'

# New targets for the three run_*.sh scripts (use direct python invocation inside container)
run_reasoner:
	$(COMPOSE_CMD) 'python src/reasoner.py --dataset "$${DATASET}" --reasoner "$${REASONER_MODEL}" --textualizer "$${TEXTUALIZER_MODEL}" --input_type "$${FLOWCHART_CODE_FORMAT}"'

run_reasoner_tool_use:
	$(COMPOSE_CMD) 'python src/reasoner.py --dataset "$${DATASET}" --reasoner "$${REASONER_MODEL}" --textualizer "$${TEXTUALIZER_MODEL}" --input_type "$${FLOWCHART_CODE_FORMAT}" --tool_use'

run_textualizer:
	$(COMPOSE_CMD) 'python src/textualizer.py --dataset "$${DATASET}" --textualizer "$${TEXTUALIZER_MODEL}" --output_type "$${FLOWCHART_CODE_FORMAT}"'

run_end_to_end:
	$(COMPOSE_CMD) 'python src/vqa.py --dataset "$${DATASET}" --model_name "$${END_TO_END_MODEL}"'

run_notebooks:
	$(COMPOSE_CMD) 'jupyter notebook --notebook-dir="/app/notebook" --ip=0.0.0.0 --NotebookApp.token="" --port=8888 --allow-root --no-browser'
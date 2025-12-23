# Device selection: 'gpu' or 'cpu' (override via `make DEVICE=cpu ...`)
DEVICE ?= gpu

ifeq ($(DEVICE),cpu)
FLOWCHART_VQA_SERVICE := flowchart_vqa_cpu
GENFLOWCHART_SERVICE := genflowchart_cpu
else
FLOWCHART_VQA_SERVICE := flowchart_vqa
GENFLOWCHART_SERVICE := genflowchart
endif

FLOWCHART_VQA_COMPOSE_CMD = docker compose run --rm $(FLOWCHART_VQA_SERVICE) bash -lc
GENFLOWCHART_COMPOSE_CMD = docker compose run --rm $(GENFLOWCHART_SERVICE) bash -lc

.PHONY: eval_textflow eval_textflow_tool_use eval_end_to_end run_reasoner run_reasoner_tool_use run_textualizer run_end_to_end
.PHONY: eval_genflowchart eval_genflowchart_cpu eval_genflowchart_gpu

eval_textflow:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/textflow/$${FLOWCHART_CODE_FORMAT}_reasoner_$${REASONER_MODEL##*/}_textualizer_$${TEXTUALIZER_MODEL##*/}.json"'

eval_textflow_tool_use:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/textflow/$${FLOWCHART_CODE_FORMAT}_reasoner_tool_use_$${REASONER_MODEL##*/}_textualizer_$${TEXTUALIZER_MODEL##*/}.json"'

eval_end_to_end:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/evaluation.py --model_names "$${JUDGE_MODELS}" --data_path "output/$${DATASET}/vqa/$${END_TO_END_MODEL##*/}.json"'

# New targets for the three run_*.sh scripts (use direct python invocation inside container)
run_reasoner:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/reasoner.py --dataset "$${DATASET}" --reasoner "$${REASONER_MODEL}" --textualizer "$${TEXTUALIZER_MODEL}" --input_type "$${FLOWCHART_CODE_FORMAT}"'

run_reasoner_tool_use:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/reasoner.py --dataset "$${DATASET}" --reasoner "$${REASONER_MODEL}" --textualizer "$${TEXTUALIZER_MODEL}" --input_type "$${FLOWCHART_CODE_FORMAT}" --tool_use'

run_textualizer:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/textualizer.py --dataset "$${DATASET}" --textualizer "$${TEXTUALIZER_MODEL}" --output_type "$${FLOWCHART_CODE_FORMAT}"'

run_end_to_end:
	$(FLOWCHART_VQA_COMPOSE_CMD) 'python src/vqa.py --dataset "$${DATASET}" --model_name "$${END_TO_END_MODEL}"'


GENFLOWCHART_OUTPUT ?= /home/zhengxin.chai/GenFlowchart/output/genflowchart_inference.json
GENFLOWCHART_INPUT ?= /home/zhengxin.chai/GenFlowchart/data/flowvqa/dev.json
GENFLOWCHART_IMAGE_DIR ?= /home/zhengxin.chai/GenFlowchart/data/images

eval_genflowchart:
    @echo "Running GenFlowchart eval (device=$(DEVICE), service=$(GENFLOWCHART_SERVICE)) -> $(GENFLOWCHART_OUTPUT)"
    $(GENFLOWCHART_COMPOSE_CMD) 'python src/main.py "$(GENFLOWCHART_INPUT)" --image-dir "$(GENFLOWCHART_IMAGE_DIR)" --output "$(GENFLOWCHART_OUTPUT)" --device $(DEVICE)'
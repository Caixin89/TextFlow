# FlowchartVQA_Research — Run guide (root project)

This README documents how to build and run the root project. It describes the Docker Compose services, how to run the evaluation / pipeline scripts, and convenient Makefile targets.

Prerequisites
- Docker (with Compose V2: use `docker compose`, not the legacy `docker-compose`).
- For GPU builds: host with NVIDIA drivers + nvidia-container-toolkit.
- On macOS (no GPU), use the CPU service (`flowchart_vqa_cpu`).

Repository layout (relevant)
- Dockerfile         — GPU (devel) image
- Dockerfile.cpu     — CPU-only image (for mac)
- docker-compose.yml
- .env               — env vars consumed by scripts (not committed)
- Makefile           — convenient targets that call docker compose
- requirements*.txt
- src/               — python code (vqa, reasoner, textualizer, evaluation)
- data/, logs/, output/ — host folders to mount (create if missing)

Required .env variables (examples)
- JUDGE_MODEL
- DATASET
- FLOWCHART_CODE_FORMAT
- REASONER_MODEL
- TEXTUALIZER_MODEL
- END_TO_END_MODEL

(Place your values in `.env` at repo root. The Makefile and compose services read this file.)

Prepare host folders
```bash
mkdir -p data logs output
# move your datasets under data/, e.g. data/genflowchart if needed
```

Build images (recommended via Compose)
- Enable BuildKit and Compose CLI for faster builds (optional but recommended):
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

- Build both images:
```bash
docker compose build
```
- Or build just one service:
```bash
docker compose build flowchart_vqa        # GPU image (Linux + CUDA hosts)
docker compose build flowchart_vqa_cpu    # CPU image (macOS / no GPU)
```

Run services
- Start nothing by default (we use `docker compose run` for one-shot commands). To run interactive containers use:
```bash
# start a shell in CPU container:
docker compose run --rm flowchart_vqa_cpu bash
# start a shell in GPU container:
docker compose run --rm --gpus all flowchart_vqa bash
```

Use the Makefile (recommended)
The Makefile provides short targets that call the appropriate Python scripts inside the container using environment variables from `.env`.

- Defaults to CPU service `flowchart_vqa_cpu`. Override with `SERVICE=flowchart_vqa` to run on GPU.
- Examples:
```bash
# run the reasoner (CPU)
make run_reasoner

# run the reasoner with tool use enabled
make run_reasoner_tool_use

# run textualizer
make run_textualizer

# run end-to-end VQA (calls src/vqa.py)
make run_end_to_end

# run evaluation: textflow / evaluation of textualization
make eval_textflow
make eval_textflow_tool_use

# run judge evaluation for end-to-end
make eval_end_to_end

# run on GPU service instead:
make SERVICE=flowchart_vqa run_reasoner
```

Direct docker-compose run
If you prefer not to use Makefile, you can run the Python commands directly. Example (uses `.env` values injected by compose):

```bash
# textflow evaluation (tool_use example)
docker compose run --rm flowchart_vqa_cpu bash -lc \
  'python src/evaluation.py --model_name "$JUDGE_MODEL" --data_path "output/${DATASET}/textflow/${FLOWCHART_CODE_FORMAT}_reasoner_tool_use_${REASONER_MODEL##*/}_textualizer_${TEXTUALIZER_MODEL##*/}.json"'

# end-to-end evaluation
docker compose run --rm flowchart_vqa_cpu bash -lc \
  'python src/evaluation.py --model_name "$JUDGE_MODEL" --data_path "output/${DATASET}/vqa/${END_TO_END_MODEL##*/}.json"'
```

Notes about flash-attn and CPU
- The CPU image excludes GPU-only packages (e.g., `flash_attn`). The CPU service sets `ENABLE_FLASH_ATTENTION=0` so code will avoid using flash-attention.
- Use `flowchart_vqa` (GPU) for best performance on Linux hosts with GPUs; use `flowchart_vqa_cpu` on macOS.

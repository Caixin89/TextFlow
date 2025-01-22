<p align="center">
    <img src="assets/figures/logo.png" width="160"> 
</p>

# Beyond End-to-End VLMs: Leveraging Intermediate Text Representations for Superior Flowchart Understanding

[![](https://img.shields.io/badge/cs.CV-arXiv%3A2412.16420-B31B1B.svg)](https://arxiv.org/abs/2412.16420)

![](./assets/figures/textflow.png)

## News
- [2025/01] 🔥 **Our TextFlow paper is accepted by [NAACL 2025](https://2025.naacl.org/)**.
- [2025/01]  The data and code for TextFlow have been released.
- [2024/12]  Excited to announce that our TextFlow paper is now available on [arXiv](https://arxiv.org/abs/2412.16420)!

## TL;DR
TEXTFLOW, a framework that converts flowchart images into text to improve explainability and control in flowchart understanding tasks.

## Abstract
Flowcharts are typically presented as images, driving the trend of using vision-language models (VLMs) for end-to-end flowchart understanding. However, two key challenges arise: **(i) Limited controllability**—users have minimal influence over the downstream task, as they can only modify input images, while the training of VLMs is often out of reach for most researchers. **(ii) Lack of explainability**—it is difficult to trace VLM errors to specific causes, such as failures in visual encoding or reasoning. 

We propose TextFlow, addressing aforementioned issues with two stages: **(i) Vision Textualizer**—which generates textual representations from flowchart images; and **(ii) Textual Reasoner**—which performs question-answering based on the text representations. TextFlow offers three key advantages: **(i) users can select the type of text representations** (e.g., Graphviz, Mermaid, PlantUML), or further convert them into **executable graph object to call tools**, enhancing performance and controllability; **(ii) it improves explainability by helping to attribute errors more clearly to visual or textual processing components**; and **(iii) it promotes the modularization of the solution**, such as allowing advanced LLMs to be used in the reasoner stage when VLMs underperform in end-to-end fashion. Experiments on the FlowVQA and FlowLearn benchmarks demonstrate TextFlow's state-of-the-art performance as well as its robustness. All code will be publicly released.

## Installation

Follow these steps to get started with **TextFlow**:

### 1. Install Dependencies and Set UP API Keys
Run the following command to install all required dependencies:
```bash
cd TextFlow
pip install -r requirements.txt
```
Set up your OpenAI or Anthropic API keys in the `config.json` file.


## Quick Start

### 1. Run the Baseline (End-to-End Visual Question Answering)
Perform baseline VQA on the `flowvqa` dataset:
```bash
python src/vqa.py --dataset flowvqa --model_name gpt-4o
```

Evaluate the experimental results:
```bash
python src/evaluation.py --model_name gpt-4o --data_path output/flowvqa/vqa/gpt-4o.json
```

---

### 2. Run the TextFlow Pipeline
#### Step 1: Vision Textualizer
Convert flowchart images into text representations (Mermaid, Graphviz, or PlantUML). Example for generating the **Mermaid** text representations:
```bash
python src/textualizer.py --dataset flowvqa --textualizer gpt-4o --output_type mermaid
```

#### Step 2: Textual Reasoner
Perform question answering based on the text representations:
```bash
python src/reasoner.py --dataset flowvqa --reasoner gpt-4o --textualizer gpt-4o --input_type mermaid
```

#### Optional: Enable Tool Use
For enhanced capabilities, enable tool usage (currently supported for Mermaid text representation with `gpt-4o`):
```bash
python src/reasoner.py --dataset flowvqa --reasoner gpt-4o --textualizer gpt-4o --input_type mermaid --tool_use
```

Evaluate the results of the TextFlow pipeline:
1. Without tool use:
   ```bash
   python evaluation.py --model_name gpt-4o --data_path output/flowvqa/textflow/mermaid_reasoner_gpt-4o_textualizer_gpt-4o.json
   ```
2. With tool use:
   ```bash
   python evaluation.py --model_name gpt-4o --data_path output/flowvqa/textflow/mermaid_reasoner_tool_use_gpt-4o_textualizer_gpt-4o.json
   ```

---

## Citation
If you find this project is helpful to your research, please consider to cite our paper:
```
@article{ye2024beyond,
  title={Beyond End-to-End VLMs: Leveraging Intermediate Text Representations for Superior Flowchart Understanding},
  author={Ye, Junyi and Dash, Ankan and Yin, Wenpeng and Wang, Guiling},
  journal={arXiv preprint arXiv:2412.16420},
  year={2024}
}
```

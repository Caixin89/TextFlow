import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from config import config
from logger import setup_logger
from models import ModelWrapper
from prompts import load_reasoner_prompt
from utils import strip_provider_from_get_model_name


def main():
    parser = argparse.ArgumentParser(description="Run the Textual Reasoner program.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="flowvqa",
        help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).",
    )
    parser.add_argument(
        "--reasoner",
        type=str,
        default="Llama-3.1-8B",
        help="The LLM to perform reasoning on the text represenation.",
    )
    parser.add_argument(
        "--textualizer",
        type=str,
        default="Qwen2-VL-7B",
        help="The VLM was used to generate the text represenation. Use 'Gound-Truth' for ground truth text representation.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="mermaid",
        help="Text representation input format (mermaid, graphviz or plantuml)",
    )
    parser.add_argument(
        "--tool_use",
        action="store_true",
        help="Whether to use tool (default: False)",
    )
    args = parser.parse_args()
    dataset = args.dataset
    reasoner = args.reasoner
    reasoner_without_provider = strip_provider_from_get_model_name(reasoner)
    textualizer = args.textualizer
    textualizer_without_provider = strip_provider_from_get_model_name(textualizer)
    input_type = args.input_type
    tool_use = args.tool_use

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tool_use:
        log_file = os.path.join(
            config["logging"]["log_dir"],
            dataset,
            f"{input_type}_reasoner_tool_use_{reasoner_without_provider}_textulizer_{textualizer_without_provider}_{timestamp}.log",
        )
    else:
        log_file = os.path.join(
            config["logging"]["log_dir"],
            dataset,
            f"{input_type}_reasoner_{reasoner_without_provider}_textulizer_{textualizer_without_provider}_{timestamp}.log",
        )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the Textural Reasoner program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    model = ModelWrapper(reasoner)

    data_path = os.path.join(config["file_paths"][dataset], "test.json")
    with open(data_path, "r") as file:
        data = json.load(file)
    keys = list(data.keys())

    # Load text represenations: Mermaid, Graphviz, PlantUML codes
    represenation_dir = os.path.join(
        config["file_paths"]["output"], dataset, input_type
    )
    represenation_file = os.path.join(represenation_dir, f"{textualizer_without_provider}.json")
    with open(represenation_file, "r") as file:
        representations = json.load(file)

    results = {}
    sample_id = 0
    for key in tqdm(keys):
        sample = data[key]
        representation = representations[key]
        question_ids = list(sample["qa"].keys())
        for question_id in question_ids:
            question = sample["qa"][question_id]["Q"]
            answer = sample["qa"][question_id]["A1"]
            prompt = load_reasoner_prompt(question, representation)

            if tool_use:
                response = model.generate_response(
                    prompt, representation=representation
                )
            else:
                response = model.generate_response(prompt)

            results[sample_id] = {
                "key": key,
                "question_id": question_id,
                "question": question,
                "response": response,
                "answer": answer,
            }
            sample_id += 1

    output_dir = os.path.join(config["file_paths"]["output"], dataset, "textflow")
    if tool_use:
        output_file = os.path.join(
            output_dir,
            f"{input_type}_reasoner_tool_use_{reasoner_without_provider}_textualizer_{textualizer_without_provider}.json",
        )
    else:
        output_file = os.path.join(
            output_dir,
            f"{input_type}_reasoner_{reasoner_without_provider}_textualizer_{textualizer_without_provider}.json",
        )
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()

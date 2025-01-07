import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from config import config
from logger import setup_logger
from models import ModelWrapper
from prompts import load_textualizer_prompt
from utils import extract_representation


def main():
    parser = argparse.ArgumentParser(
        description="Run the Vision Textualizer program. (Convert flowchart to text Representation)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="flowvqa",
        help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).",
    )
    parser.add_argument(
        "--textualizer",
        type=str,
        default="Qwen2-VL-7B",
        help="The VLM to generate the text represenation.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="mermaid",
        help="Text representation output format (mermaid, graphviz or plantuml)",
    )
    args = parser.parse_args()
    dataset = args.dataset
    textualizer = args.textualizer
    output_type = args.output_type

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        dataset,
        f"{output_type}_textulizer_{textualizer}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the Vision Textualizer program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    model = ModelWrapper(textualizer)

    data_path = os.path.join(config["file_paths"][dataset], "test.json")
    with open(data_path, "r") as file:
        data = json.load(file)
    keys = list(data.keys())

    results = {}
    image_extension = "jpeg" if dataset == "flowlearn" else "png"
    for key in tqdm(keys):
        image_path = os.path.join(
            config["file_paths"][dataset], "images", f"{key}.{image_extension}"
        )
        prompt = load_textualizer_prompt(output_type)
        response = model.generate_response(prompt, image_path=image_path)
        # Extract text representation (mermaid, graphviz, or plantuml) from response
        representation = extract_representation(response)
        results[key] = representation

    output_dir = os.path.join(config["file_paths"]["output"], dataset, output_type)
    output_file = os.path.join(output_dir, f"{textualizer}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()

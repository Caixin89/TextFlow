import argparse
import json
import logging
import os
from datetime import datetime

from PIL import Image
from tqdm import tqdm

from config import config
from logger import setup_logger
from models import generate_response, load_model_and_tokenizer
from prompts import load_vqa_prompt
from utils import encode_image_anthropic, encode_image_openai


def main():
    parser = argparse.ArgumentParser(
        description="Run the Vision Question Answering (VQA) program."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="flowvqa",
        help="Dataset to use (flowvqa or flowlearn).",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen2-VL-7B", help="The VLM to perform VQA."
    )
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"], dataset, f"vqa_{model_name}_{timestamp}.log"
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the Vision Question Answering program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    model, processor = load_model_and_tokenizer(model_name)

    data_path = os.path.join(config["file_paths"][dataset], "test.json")
    with open(data_path, "r") as file:
        data = json.load(file)
    keys = list(data.keys())

    results = {}
    sample_id = 0
    image_extension = "jpeg" if dataset == "flowlearn" else "png"
    for key in tqdm(keys):
        sample = data[key]
        image_file = os.path.join(
            config["file_paths"][dataset], "images", f"{key}.{image_extension}"
        )
        if model_name == "claude-3-5-sonnet":
            image = encode_image_anthropic(image_file)
        elif model_name in ["gpt-4o", "gpt-4o-mini"]:
            image = encode_image_openai(image_file)
        else:
            image = Image.open(image_file)
        question_ids = list(sample["qa"].keys())
        for question_id in question_ids:
            question = sample["qa"][question_id]["Q"]
            answer = sample["qa"][question_id]["A1"]
            prompt = load_vqa_prompt(question)
            response = generate_response(model_name, model, processor, prompt, image)
            results[sample_id] = {
                "key": key,
                "question_id": question_id,
                "question": question,
                "response": response,
                "answer": answer,
            }
            sample_id += 1

    output_dir = os.path.join(config["file_paths"]["output"], dataset, "vqa")
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()

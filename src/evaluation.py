import argparse
import json
import logging
import os
import re
from datetime import datetime

from tqdm import tqdm

from config import config
from logger import setup_logger
from models import ModelWrapper
from prompts import load_evaluation_prompt
from utils import majority_vote


def main():
    parser = argparse.ArgumentParser(
        description="Run the Question Answering Evaluation program."
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "openrouter/openai/gpt-5.1-chat",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/mistralai/mistral-large-2411",
        ],
        help=(
            "One or more model names (e.g. --model_names a b c), "
            "or a single comma- or space-separated string (e.g. \"a,b,c\" or \"a b c\"). "
            "The value will be normalized into a Python list of strings."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=888,
        help="Random seed",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/flowvqa/textflow/mermaid_reasoner_Llama-3.1-8B_textualizer_Qwen2-VL-7B.json",
        help="Data path of the experiment result to evaluate.",
    )
    args = parser.parse_args()

    # Normalize model_names into a flat list[str].
    # Accepts: separate args, comma-separated single arg, or space-separated single arg from env.
    model_names = []
    for token in args.model_names:
        model_names.extend([m for m in re.split(r'[,\s]+', token.strip()) if m])
    args.model_names = model_names

    model_names = args.model_names
    seed = args.seed
    data_path = args.data_path
    dataset = data_path.split("/")[2]
    exp_dir = os.path.dirname(data_path)
    exp_name = os.path.splitext(os.path.basename(data_path))[0]

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        dataset,
        f"evaluation_{exp_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting the Question Answering Evaluation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    models = [ModelWrapper(m) for m in model_names]

    with open(data_path, "r") as file:
        data = json.load(file)

    for key, sample in tqdm(data.items()):
        prompt = load_evaluation_prompt(
            sample["question"], sample["response"], sample["answer"]
        )
        judgements = [m.generate_evaluation_response(prompt, seed) for m in models]
        final_decision = majority_vote([j["verdict"] for j in judgements])
        result = {"final_decision": final_decision}
        for j in judgements:
            result[f"decision{len(result)+1}"] = { "verdict": j["verdict"], "explanation": j["explanation"] }
        # Append the evaluation result to
        data[key] = {**data[key], **result}

    with open(data_path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(data_path)}")

    # Calculate accuracy
    correct_count = 0
    total_count = len(data)

    for sample in data.values():
        if sample["final_decision"] == "Correct":
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

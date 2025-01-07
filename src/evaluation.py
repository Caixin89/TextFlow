import argparse
import json
import logging
import os
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
        "--model_name",
        type=str,
        default="gpt-4o",
        help="The LLM used as the evaluator.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/flowvqa/textflow/mermaid_reasoner_Llama-3.1-8B_textualizer_Qwen2-VL-7B.json",
        help="Data path of the experiment result to evaluate.",
    )
    args = parser.parse_args()
    model_name = args.model_name
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

    model = ModelWrapper(model_name)

    with open(data_path, "r") as file:
        data = json.load(file)

    for key, sample in tqdm(data.items()):
        prompt = load_evaluation_prompt(
            sample["question"], sample["response"], sample["answer"]
        )
        decision1, decision2, decision3 = model.generate_evaluation_response(prompt)
        final_decision = majority_vote(decision1, decision2, decision3)
        result = {
            "decision1": decision1,
            "decision2": decision2,
            "decision3": decision3,
            "final_decision": final_decision,
        }
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

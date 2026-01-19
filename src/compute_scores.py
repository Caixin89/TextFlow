#!/usr/bin/env python3
"""
Compute accuracy breakdowns from:
  1) an evaluation result JSON file
  2) the original dataset JSON file

Outputs 8 scores (rounded to 2 decimal places):
- overall accuracy
- accuracy in code subset
- accuracy in instruct subset
- accuracy in wiki subset
- accuracy in questions of T1 category
- accuracy in questions of T2 category
- accuracy in questions of T3 category
- accuracy in questions of T4 category
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# -----------------------------
# I/O utilities
# -----------------------------

def load_json(path: Path) -> Any:
    """Load JSON from disk with friendly errors."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON in {path}: {e}")


def fmt(x: float) -> str:
    """Format float to 2 decimal places."""
    return f"{x:.2f}"


# -----------------------------
# Score computation placeholders
# -----------------------------

def compute_overall_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute overall accuracy."""
    return sum(v["final_decision"].lower()=="correct" for v in eval_data.values()) / len(eval_data)


def compute_code_subset_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for code subset."""
    rows = [v for v in eval_data.values() if v["key"].startswith("code")]
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_instruct_subset_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for instruct subset."""
    rows = [v for v in eval_data.values() if v["key"].startswith("instruct")]
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_wiki_subset_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for wiki subset."""
    rows = [v for v in eval_data.values() if v["key"].startswith("wiki")]
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_T1_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for T1 category."""
    rows = []
    for v in eval_data.values():
        question_id = v["question_id"]
        key = v["key"]
        if dataset_data[key]["qa"][question_id]["type"] == "fact_retrieval":
            rows.append(v)
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_T2_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for T2 category."""
    rows = []
    for v in eval_data.values():
        question_id = v["question_id"]
        key = v["key"]
        if dataset_data[key]["qa"][question_id]["type"] == "applied_scenario":
            rows.append(v)
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_T3_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for T3 category."""
    rows = []
    for v in eval_data.values():
        question_id = v["question_id"]
        key = v["key"]
        if dataset_data[key]["qa"][question_id]["type"] == "flow_referential":
            rows.append(v)
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


def compute_T4_accuracy(eval_data: Any, dataset_data: Any) -> float:
    """Compute accuracy for T4 category."""
    rows = []
    for v in eval_data.values():
        question_id = v["question_id"]
        key = v["key"]
        if dataset_data[key]["qa"][question_id]["type"] == "topological":
            rows.append(v)
    return sum(v["final_decision"].lower()=="correct" for v in rows) / len(rows)


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score_breakdown.py",
        description=(
            "Compute accuracy breakdowns from an evaluation result JSON file and the original dataset JSON file.\n\n"
            "Outputs 8 scores (rounded to 2 decimals):\n"
            "  - overall\n"
            "  - code / instruct / wiki subsets\n"
            "  - T1 / T2 / T3 / T4 categories"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--eval",
        required=True,
        type=Path,
        help="Path to evaluation result JSON file.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to original dataset JSON file.",
    )
    parser.add_argument(
        "--as-percent",
        action="store_true",
        help=(
            "Interpret returned scores as fractions in [0,1] and print as percentages.\n"
            "Do NOT set this flag if your functions already return percentages."
        ),
    )

    return parser.parse_args(argv)


# -----------------------------
# Main
# -----------------------------

def main(argv: list[str]) -> int:
    args = parse_args(argv)

    eval_data = load_json(args.eval)
    dataset_data = load_json(args.dataset)

    try:
        scores = {
            "overall_accuracy": compute_overall_accuracy(eval_data, dataset_data),
            "accuracy_code_subset": compute_code_subset_accuracy(eval_data, dataset_data),
            "accuracy_instruct_subset": compute_instruct_subset_accuracy(eval_data, dataset_data),
            "accuracy_wiki_subset": compute_wiki_subset_accuracy(eval_data, dataset_data),
            "accuracy_T1": compute_T1_accuracy(eval_data, dataset_data),
            "accuracy_T2": compute_T2_accuracy(eval_data, dataset_data),
            "accuracy_T3": compute_T3_accuracy(eval_data, dataset_data),
            "accuracy_T4": compute_T4_accuracy(eval_data, dataset_data),
        }
    except NotImplementedError:
        print("ERROR: One or more score computation functions are not implemented.", file=sys.stderr)
        return 2

    if args.as_percent:
        scores = {k: v * 100.0 for k, v in scores.items()}

    order = [
        "overall_accuracy",
        "accuracy_code_subset",
        "accuracy_instruct_subset",
        "accuracy_wiki_subset",
        "accuracy_T1",
        "accuracy_T2",
        "accuracy_T3",
        "accuracy_T4",
    ]

    for key in order:
        print(f"{key}: {fmt(scores[key])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

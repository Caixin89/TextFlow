import argparse
import json
import os
import random
from typing import Tuple, Any

def split_items(items: list, dev_ratio: float, seed: int) -> Tuple[list, list]:
    rng = random.Random(seed)
    rng.shuffle(items)
    n_dev = int(len(items) * dev_ratio)
    # Ensure at least one item in each split when possible
    if len(items) > 1:
        n_dev = max(1, min(n_dev, len(items) - 1))
    return items[:n_dev], items[n_dev:]

'''Split the FlowVQA dataset into dev and test sets, where instruct, wiki and code samples are each distributed into the dev and test sets in the same ratio'''
def split_data(data: list, dev_ratio: float, seed: int) -> Tuple[list, list]:
    wiki_items = []
    instruct_items = []
    code_items = []

    for k,v in data.items():
        if k.startswith("wiki"):
            wiki_items.append((k,v))
        elif k.startswith("instruct"):
            instruct_items.append((k,v))
        elif k.startswith("code"):
            code_items.append((k,v))

    wiki_dev, wiki_test = split_items(wiki_items, dev_ratio, seed)
    instruct_dev, instruct_test = split_items(instruct_items, dev_ratio, seed)
    code_dev, code_test = split_items(code_items, dev_ratio, seed)
    rng = random.Random(seed)
    dev = dict(rng.shuffle(wiki_dev + instruct_dev + code_dev))
    rng = random.Random(seed)
    test = dict(rng.shuffle(wiki_test + instruct_test + code_test))
    return dev, test

def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Split a JSON QA dataset into dev and test sets.")
    parser.add_argument("input", help="Path to input JSON file (list or dict).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Fraction of examples to place in dev (0.0-1.0, default 0.2).")
    parser.add_argument("--out-dev", help="Output path for dev JSON (default: <input>_dev.json).")
    parser.add_argument("--out-test", help="Output path for test JSON (default: <input>_test.json).")
    args = parser.parse_args()

    if not 0.0 <= args.dev_ratio <= 1.0:
        parser.error("--dev-ratio must be between 0.0 and 1.0")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = os.path.splitext(args.input)[0]
    out_dev = args.out_dev or f"{base}_dev.json"
    out_test = args.out_test or f"{base}_test.json"

    items = list(data.items())  # list of (key, value)
    dev_items, test_items = split_items(items, args.dev_ratio, args.seed)
    dev = dict(dev_items)
    test = dict(test_items)
    write_json(out_dev, dev)
    write_json(out_test, test)

if __name__ == "__main__":
    main()

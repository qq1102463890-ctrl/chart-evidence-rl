"""Replay all published test responses with Python's standard library."""

import gzip
import json
from collections import Counter
from pathlib import Path

from chart_eval.content import score
from chart_eval.parsing import extract
from chart_eval.sorting import sort_targets

ROOT = Path(__file__).resolve().parent
MODELS = ("sft", "seed1", "seed2")


def replay(row, model):
    """Score model-generated order and deterministic order against the same gold."""
    readings, ranking, _ = extract(row["responses"][model])
    derived, _ = sort_targets(readings, ranking, row["targets"])
    gold = row["gold"]

    def evaluate_order(order):
        return score(json.dumps({"readings": readings, "ranking": order}), gold)

    direct = evaluate_order(ranking)
    sorted_answer = evaluate_order(derived)
    if direct["per_target"] != sorted_answer["per_target"]:
        raise ValueError(f"Reordering changed readings for {row['qid']}")
    return direct["count"], direct["both"], sorted_answer["both"]


def main():
    expected = json.loads((ROOT / "data/expected.json").read_bytes())
    print("Dataset  Model   Correct readings    Direct exact    Sorted exact    Sorted vs SFT")
    for dataset in ("plotqa", "chartqa"):
        totals = {model: Counter() for model in MODELS}
        ids = set()
        with gzip.open(ROOT / f"data/{dataset}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["qid"] in ids:
                    raise ValueError(f"Duplicate question: {row['qid']}")
                ids.add(row["qid"])
                for model in MODELS:
                    correct, direct, sorted_exact = replay(row, model)
                    totals[model].update(questions=1, targets=len(row["targets"]),
                                         correct=correct, direct=direct, sorted=sorted_exact)
        for model, counts in totals.items():
            if dict(counts) != expected[dataset][model]:
                raise ValueError(f"{dataset}/{model} differs from the archived result: {counts}")
            n = counts["questions"]
            readings = f"{counts['correct']}/{counts['targets']}"
            direct = f"{counts['direct']}/{n}"
            sorted_exact = f"{counts['sorted']}/{n}"
            delta = counts["sorted"] - totals["sft"]["sorted"]
            print(f"{dataset:8} {model:7} {readings:19} {direct:15} {sorted_exact:15} {delta:+d}")


if __name__ == "__main__":
    main()

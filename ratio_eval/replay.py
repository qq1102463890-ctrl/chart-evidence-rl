"""Recompute explicit-ratio test metrics from raw responses using the project scorer."""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chart_eval.content import score
from chart_eval.parsing import extract


def main():
    expected = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    for dataset in ("plotqa", "benetech"):
        with gzip.open(HERE / f"{dataset}.jsonl.gz", "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len({r["qid"] for r in rows}) == len(rows)
        for model in ("base", "sft", "rl"):
            correct = direct = targets = 0
            for row in rows:
                readings, ranking, _ = extract(row["responses"][model])
                result = score(json.dumps({"readings": readings, "ranking": ranking}), row["gold"])
                correct += result["count"]
                direct += int(result["both"])
                targets += len(row["targets"])
            counts = dict(n=len(rows), targets=targets, read=correct, direct=direct)
            if counts != expected[f"{model}/{dataset}"]:
                raise ValueError(f"Unexpected total for {dataset}/{model}: {counts}")
            print(f"{dataset:10} {model:5}  readings {correct}/{targets}  direct {direct}/{len(rows)}")


if __name__ == "__main__":
    main()

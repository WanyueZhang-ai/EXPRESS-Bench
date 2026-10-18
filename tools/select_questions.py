"""Save a reproducible subset for reuse across benchmark conditions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.question_selection import load_question_count, select_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/express-bench.json")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    indices = select_indices(load_question_count(args.dataset), sample_size=args.sample_size, sample_seed=args.sample_seed)
    output = Path(args.output_file).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(indices, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": len(indices), "sample_seed": args.sample_seed, "indices": indices, "output_file": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

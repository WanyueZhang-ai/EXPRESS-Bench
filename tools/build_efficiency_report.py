"""Build/merge efficiency reports without loading models or Habitat."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.efficiency_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="one run directory, or worker shard directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--merge-traces", action="store_true")
    args = parser.parse_args()
    summary = build_report(args.inputs, args.output_dir, merge_traces=args.merge_traces)
    print(json.dumps({"episodes": summary["episodes"]["count"], "steps": summary["steps"]["count"], "calls": summary["calls"]["count"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

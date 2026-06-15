"""CLI: score a localization jsonl against gold at a chosen level.

Works on any jsonl whose rows carry ``instance_id`` + ``found_files`` / ``found_related_locs`` /
``found_edit_locs`` — i.e. ECR-HiFL pipeline results, upstream HiLoRM ``loc_outputs.jsonl`` (its
own pre-selected pick), or a baseline dump. Useful both to report ECR results and to cross-check
parity with ``eval/step1-jisuan.py`` etc.

  python -m ecr_hifl.eval.evaluate --level file --pred results/.../loc_outputs.jsonl \
                                   --gold data/modified_files.jsonl
"""

from __future__ import annotations

import argparse

from ..config import ECRConfig
from ..io_utils import load_jsonl_by_id
from .metrics import compute_metrics, format_metrics, load_gold


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a localization jsonl vs gold")
    p.add_argument("--level", required=True, choices=["file", "function", "line"])
    p.add_argument("--pred", required=True, help="results/pool jsonl with found_* fields")
    p.add_argument("--gold", help="gold path (defaults to the config's gold for this level)")
    p.add_argument("--config", help="configs/ecr_*.yaml (for default gold path)")
    args = p.parse_args(argv)

    cfg = ECRConfig.load(args.config)
    gold_path = args.gold or cfg.gold_path(args.level)
    preds = load_jsonl_by_id(args.pred)
    gold = load_gold(args.level, gold_path)
    m = compute_metrics(args.level, preds, gold)
    print(f"pred={args.pred}")
    print(f"gold={gold_path}")
    print(format_metrics(m))
    return m


if __name__ == "__main__":
    main()

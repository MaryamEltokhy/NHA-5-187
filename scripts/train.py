"""Train a segmentation experiment and evaluate it on the validation patients (M2-T03 and later experiments).

    python scripts/train.py --exp configs/training/EXP-0001.yaml --max-hours 3.5

Rerunning the same command resumes from the last checkpoint in output_dir (same MLflow run).
Add --eval-only to re-run the final evaluation of best.pt. Locked test patients are never loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmts.segmentation.training.trainer import final_evaluation, load_experiment, mirror_tracking_db, train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="configs/training/EXP-0001.yaml", help="experiment config")
    p.add_argument("--processed-root", help="override the cleaned-scans folder")
    p.add_argument("--split-csv", help="override the split file")
    p.add_argument("--output-dir", help="override where checkpoints and final metrics go")
    p.add_argument("--tracking-uri", help="override the MLflow tracking URI")
    p.add_argument("--epochs", type=int)
    p.add_argument("--iters-per-epoch", type=int)
    p.add_argument("--val-every", type=int)
    p.add_argument("--train-cases", type=int, help="use only the first N training patients (quick tests)")
    p.add_argument("--val-cases", type=int, help="final evaluation on the first N validation patients only")
    p.add_argument("--val-cases-during-training", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--smart-cache-num", type=int, help="patients kept in RAM (0 = read from disk every time)")
    p.add_argument("--max-hours", type=float, help="stop cleanly after this many hours; rerun to resume")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt and start over")
    p.add_argument("--eval-only", action="store_true", help="only run the final evaluation of best.pt")
    args = p.parse_args()

    cfg = load_experiment(args.exp, processed_root=args.processed_root, split_csv=args.split_csv,
                          output_dir=args.output_dir, tracking_uri=args.tracking_uri, epochs=args.epochs,
                          iters_per_epoch=args.iters_per_epoch, val_every=args.val_every, train_cases=args.train_cases,
                          val_cases=args.val_cases, val_cases_during_training=args.val_cases_during_training,
                          num_workers=args.num_workers, smart_cache_num=args.smart_cache_num)
    if args.eval_only:
        import mlflow
        result = final_evaluation(cfg)
        mlflow.end_run()
        mirror_tracking_db(cfg)
    else:
        result = train(cfg, max_hours=args.max_hours, resume=not args.no_resume)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("finished", True):
        print("NOT FINISHED YET: run the same command again to continue from the last checkpoint.")
    else:
        print("TRAINING FINISHED: validation scores are logged in MLflow and saved in", cfg["exp"]["output_dir"])


if __name__ == "__main__":
    main()

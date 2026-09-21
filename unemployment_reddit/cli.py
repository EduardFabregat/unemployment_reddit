"""Command line entry point: ``python -m unemployment_reddit <stage>``."""

import argparse

from . import pipeline
from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unemployment_reddit", description=__doc__)
    parser.add_argument("--data-dir", help="folder for the database and artifacts (env REDDIT_DATA_DIR)")
    parser.add_argument("--db", help="SQLite database path (env REDDIT_DB_PATH)")
    parser.add_argument("--tag", help="suffix added to artifact names, e.g. a run date")
    stages = parser.add_subparsers(dest="stage", required=True)

    ingest = stages.add_parser("ingest", help="stage 1: load .zst dumps into SQLite")
    ingest.add_argument("--submissions", help="path to the submissions .zst file")
    ingest.add_argument("--comments", help="path to the comments .zst file")

    stages.add_parser("topics", help="stage 2: fit BERTopic on a sample")

    assign = stages.add_parser("assign", help="stage 3: assign all documents, then emotions")
    assign.add_argument("--skip-emotions", action="store_true")
    assign.add_argument("--emotion-batch-size", type=int, help="batch size for emotion classification")

    analyze = stages.add_parser("analyze", help="stage 4: thread features, CEC and models")
    analyze.add_argument("--xgb-iter", type=int, default=20)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    overrides = {"data_dir": args.data_dir, "db_path": args.db, "tag": args.tag}
    cfg = Config(**{k: v for k, v in overrides.items() if v is not None})

    if args.stage == "ingest":
        pipeline.run_ingest(cfg, args.submissions, args.comments)
    elif args.stage == "topics":
        pipeline.run_topics(cfg)
    elif args.stage == "assign":
        emotion_kwargs = {}
        if args.emotion_batch_size:
            emotion_kwargs["batch_size"] = args.emotion_batch_size
        pipeline.run_assign(cfg, with_emotions=not args.skip_emotions, **emotion_kwargs)
    elif args.stage == "analyze":
        pipeline.run_analysis(cfg, xgb_iter=args.xgb_iter)

"""Command line entry points. No experiments run at import time."""
import argparse
import os
import logging


def main():
    # CLI processes render files and never need Colab's notebook-only backend.
    os.environ["MPLBACKEND"] = "Agg"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .task1 import METHODS
    from .task1_eval import CONDITIONS
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.set_defaults(func=lambda a: __import__("atml_pa3.doctor", fromlist=["run"]).run(a))
    task = commands.add_parser("task1")
    sub = task.add_subparsers(dest="action", required=True)
    from . import task1, task1_eval, evidence
    sub.add_parser("prepare").set_defaults(func=task1.prepare)
    train = sub.add_parser("train")
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--resume", help="Trusted local recovery checkpoint")
    train.set_defaults(func=task1.train)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--condition", choices=CONDITIONS, required=True)
    evaluate.add_argument("--batch-size", type=int, default=2)
    evaluate.set_defaults(func=task1_eval.evaluate)
    sub.add_parser("diagnostics").set_defaults(func=task1_eval.diagnostics)
    qualitative = sub.add_parser("qualitative")
    from .common import ROOT
    qualitative.add_argument("--prompts", default=str(ROOT / "configs/first_plot_questions.yaml"))
    qualitative.add_argument("--condition", choices=CONDITIONS, help="Generate one condition; combine after all seven")
    qualitative.add_argument("--batch-size", type=int, default=1)
    qualitative.set_defaults(func=task1_eval.qualitative)
    sub.add_parser("summarize").set_defaults(func=evidence.build)
    sub.add_parser("audit").set_defaults(func=evidence.audit)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

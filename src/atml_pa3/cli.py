import argparse
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def handler(module, name):
    def invoke(args):
        return getattr(import_module(f"atml_pa3.{module}"), name)(args)

    return invoke


def main():
    parser = argparse.ArgumentParser(
        description="ATML PA3 reproducible experiment commands"
    )
    subs = parser.add_subparsers(dest="task", required=True)
    env = subs.add_parser("doctor")
    env.add_argument("--require-cuda", action="store_true")
    env.set_defaults(func=handler("doctor", "doctor"))
    t1 = subs.add_parser("task1").add_subparsers(dest="action", required=True)

    p = t1.add_parser("prepare")
    p.set_defaults(func=handler("task1", "prepare"))
    p = t1.add_parser("train")
    p.add_argument(
        "--method",
        required=True,
        choices=("full", "l2sp", "lora4", "lora16", "oplora16"),
    )
    p.add_argument("--resume")
    p.set_defaults(func=handler("task1", "train"))
    p = t1.add_parser("evaluate")
    p.add_argument(
        "--condition",
        required=True,
        choices=("baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"),
    )
    p.set_defaults(func=handler("task1_eval", "evaluate"))
    p = t1.add_parser("diagnostics")
    p.set_defaults(func=handler("task1_eval", "diagnostics"))
    p = t1.add_parser("qualitative")
    p.add_argument("--prompts", required=True)
    p.set_defaults(func=handler("task1_eval", "qualitative"))
    t2 = subs.add_parser("task2").add_subparsers(dest="action", required=True)
    p = t2.add_parser("prepare")
    p.add_argument("--annotations", required=True)
    p.set_defaults(func=handler("vlm_data", "prepare"))
    p = t2.add_parser("cache")
    p.add_argument("--images", default=str(ROOT / "data/external/coco/images"))
    p.add_argument("--download", action="store_true")
    p.set_defaults(func=handler("vlm_data", "cache"))
    p = t2.add_parser("train")
    p.add_argument("--visual-tokens", type=int, choices=[196, 49], default=196)
    p.add_argument("--resume")
    p.set_defaults(func=handler("task2", "train"))
    p = t2.add_parser("evaluate")
    p.add_argument("--visual-tokens", type=int, choices=[196, 49], default=196)
    p.set_defaults(func=handler("task2", "evaluate"))
    t3 = subs.add_parser("task3").add_subparsers(dest="action", required=True)
    p = t3.add_parser("offline")
    p.add_argument("--math-dir", required=True)
    p.add_argument("--mathvista-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--readme")
    p.add_argument("--schema", default=str(ROOT / "configs/rollout_schema.json"))
    p.set_defaults(func=handler("task3", "run"))
    p = t3.add_parser("live")
    p.add_argument("--revision", required=True)
    p.add_argument("--readme", required=True)
    p.set_defaults(func=handler("live", "run"))
    p = subs.add_parser("evidence")
    p.set_defaults(func=handler("evidence", "build"))
    p = subs.add_parser("audit")
    p.set_defaults(func=handler("evidence", "audit"))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

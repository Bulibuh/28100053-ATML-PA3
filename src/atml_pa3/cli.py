import argparse
import json
from .common import ROOT, hardware


def main():
    parser = argparse.ArgumentParser(
        description="ATML PA3 reproducible experiment commands"
    )
    subs = parser.add_subparsers(dest="task", required=True)
    env = subs.add_parser("doctor")
    env.set_defaults(func=lambda _: print(json.dumps(hardware(), indent=2)))
    t1 = subs.add_parser("task1").add_subparsers(dest="action", required=True)
    from . import task1, task1_eval, vlm_data, task2, task3, live, evidence

    p = t1.add_parser("prepare")
    p.set_defaults(func=task1.prepare)
    p = t1.add_parser("train")
    p.add_argument("--method", required=True, choices=task1.METHODS)
    p.add_argument("--resume")
    p.set_defaults(func=task1.train)
    p = t1.add_parser("evaluate")
    p.add_argument("--condition", required=True, choices=task1_eval.CONDITIONS)
    p.set_defaults(func=task1_eval.evaluate)
    p = t1.add_parser("diagnostics")
    p.set_defaults(func=task1_eval.diagnostics)
    p = t1.add_parser("qualitative")
    p.add_argument("--prompts", required=True)
    p.set_defaults(func=task1_eval.qualitative)
    t2 = subs.add_parser("task2").add_subparsers(dest="action", required=True)
    p = t2.add_parser("prepare")
    p.add_argument("--annotations", required=True)
    p.set_defaults(func=vlm_data.prepare)
    p = t2.add_parser("cache")
    p.add_argument("--images", default=str(ROOT / "data/external/coco/images"))
    p.add_argument("--download", action="store_true")
    p.set_defaults(func=vlm_data.cache)
    p = t2.add_parser("train")
    p.add_argument("--visual-tokens", type=int, choices=[196, 49], default=196)
    p.add_argument("--resume")
    p.set_defaults(func=task2.train)
    p = t2.add_parser("evaluate")
    p.add_argument("--visual-tokens", type=int, choices=[196, 49], default=196)
    p.set_defaults(func=task2.evaluate)
    t3 = subs.add_parser("task3").add_subparsers(dest="action", required=True)
    p = t3.add_parser("offline")
    p.add_argument("--math-dir", required=True)
    p.add_argument("--mathvista-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--readme")
    p.add_argument("--schema", default=str(ROOT / "configs/rollout_schema.json"))
    p.set_defaults(func=task3.run)
    p = t3.add_parser("live")
    p.add_argument("--revision", required=True)
    p.add_argument("--readme", required=True)
    p.set_defaults(func=live.run)
    p = subs.add_parser("evidence")
    p.set_defaults(func=evidence.build)
    p = subs.add_parser("audit")
    p.set_defaults(func=evidence.audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

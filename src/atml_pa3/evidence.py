"""Build tables/plots only from measured artifacts; audit missing experiments explicitly."""

import json
from pathlib import Path
import pandas as pd
from .common import *


def build(args=None):
    # Set before importing matplotlib: an invalid inherited backend fails at import.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = ARTIFACT_ROOT / "evidence"
    # Task 1 learning/resource/benchmark tables.
    methods = ["full", "l2sp", "lora4", "lora16", "oplora16"]
    fig, ax = plt.subplots(figsize=(8, 4))
    count = 0
    resources = []
    for method in methods:
        path = ev / f"task1/{method}/train.jsonl"
        if path.exists():
            df = pd.read_json(path, lines=True)
            ax.plot(
                df["update"],
                df.sft_loss.rolling(20, min_periods=1).mean(),
                label=method,
            )
            count += 1
        path = ev / f"task1/{method}/resources.json"
        if path.exists():
            resource = dict(method=method, **json.loads(path.read_text()))
            ck = ev / f"task1/{method}/checkpoints.jsonl"
            if ck.exists():
                checkpoints = read_jsonl(ck)
                resource["checkpoint_bytes_total"] = sum(
                    r["bytes"] for r in checkpoints
                )
                resource["final_checkpoint_bytes"] = checkpoints[-1]["bytes"]
            resources.append(resource)
    ax.set(xlabel="Optimizer update", ylabel="SFT response-token loss")
    ax.legend() if count else None
    if count:
        fig.tight_layout()
        fig.savefig(ev / "task1/learning_curves.png", dpi=160)
    plt.close(fig)
    if resources:
        pd.DataFrame(resources).to_csv(ev / "task1/resources.csv", index=False)
    regularization = []
    for method in ["full", "l2sp"]:
        path = ev / f"task1/{method}/checkpoints.jsonl"
        if path.exists():
            regularization.extend(
                dict(method=method, **r)
                for r in read_jsonl(path)
                if "displacement" in r
            )
    if regularization:
        pd.DataFrame(regularization).to_csv(ev / "task1/displacements.csv", index=False)
    path = ev / "task1/l2sp/train.jsonl"
    if path.exists():
        df = pd.read_json(path, lines=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(df["update"], df.sft_loss, label="SFT")
        axes[0].plot(df["update"], df.weighted_penalty, label="lambda * Omega")
        axes[0].legend()
        axes[1].plot(df["update"], df.omega)
        axes[1].set_ylabel("Unaveraged Omega")
        for ax in axes:
            ax.set_xlabel("Optimizer update")
        fig.tight_layout()
        fig.savefig(ev / "task1/regularization.png", dpi=160)
        plt.close(fig)
    results = []
    for path in (ev / "task1/evaluation").glob("*/scores.json"):
        scores = json.loads(path.read_text())
        for task, metrics in scores.items():
            for name, value in metrics.items():
                if isinstance(value, (float, int)) and "stderr" not in name:
                    results.append(
                        dict(
                            condition=path.parent.name,
                            task=task,
                            metric=name,
                            value=value,
                        )
                    )
    if results:
        df = pd.DataFrame(results)
        base = df[df.condition == "baseline"][["task", "metric", "value"]].rename(
            columns={"value": "baseline"}
        )
        df = df.merge(base, on=["task", "metric"], how="left")
        df["delta_percentage_points"] = 100 * (df.value - df.baseline)
        df.to_csv(ev / "task1/benchmark_comparison.csv", index=False)
        primary = df[
            ((df.task == "gsm8k_cot") & df.metric.str.contains("exact_match,flexible"))
            | ((df.task == "boolq") & df.metric.str.startswith("acc,"))
            | (
                df.task.isin(["hellaswag", "arc_easy"])
                & df.metric.str.startswith("acc_norm,")
            )
        ]
        primary.pivot(index="condition", columns="task", values="value").to_csv(
            ev / "task1/primary_benchmarks.csv"
        )
        if len(primary):
            from .task1_eval import CONDITIONS
            score = primary.pivot(index="condition", columns="task", values="value") * 100
            change = primary.pivot(index="condition", columns="task", values="delta_percentage_points")
            table = pd.DataFrame(index=list(CONDITIONS))
            for task in ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]:
                if task in score:
                    table[f"{task}_percent"] = score[task]
                    table[f"{task}_delta_pp"] = change[task]
            table.index.name = "condition"
            table.to_csv(ev / "task1/benchmark_table.csv")
            fig, ax = plt.subplots(figsize=(10, 4))
            primary.pivot(index="condition", columns="task", values="value").plot.bar(
                ax=ax
            )
            ax.set(ylabel="Accuracy (fraction)", xlabel="Evaluation condition")
            fig.tight_layout()
            fig.savefig(ev / "task1/benchmarks.png", dpi=160)
            plt.close(fig)
    path = ev / "task1/diagnostics/metrics.json"
    if path.exists():
        df = pd.read_json(path)
        df.groupby(["condition", "layer", "dataset"])[
            ["rho", "eta", "R"]
        ].mean().to_csv(ev / "task1/diagnostics/summary.csv")
    audit()


def audit(args=None):
    from .task1_eval import CONDITIONS
    from .task1 import METHODS
    required = ["data.json", "learning_curves.png", "benchmark_comparison.csv",
                "primary_benchmarks.csv", "resources.csv", "regularization.png",
                "displacements.csv", "diagnostics/metrics.json",
                "qualitative/responses.json", "qualitative/annotations_to_complete.csv"]
    required += [f"{m}/resources.json" for m in METHODS]
    required += [f"evaluation/{c}/scores.json" for c in CONDITIONS]
    root = ARTIFACT_ROOT / "evidence/task1"
    missing = [p for p in required if not (root / p).exists()]
    checks = {}
    qual = root / "qualitative/responses.json"
    if qual.exists():
        rows = json.loads(qual.read_text())
        checks["qualitative_168_responses"] = len(rows) == 168
        checks["annotation_subset_42"] = sum(r["annotation_required"] for r in rows) == 42
    annotations = root / "qualitative/annotations_to_complete.csv"
    if annotations.exists():
        df = pd.read_csv(annotations)
        cols = ["format_compliance", "coherence_relevance", "repetition", "math_intrusion",
                "inappropriate_content", "excerpt", "human_notes"]
        checks["human_annotations_complete"] = len(df) == 42 and df[cols].notna().all().all()
    for m in METHODS:
        log = root / m / "train.jsonl"
        if log.exists():
            records = read_jsonl(log)
            checks[f"{m}_1250_updates"] = [r["update"] for r in records] == list(range(1, 1251))
        path = root / m / "checkpoints.jsonl"
        if path.exists():
            checks[f"{m}_recovery_checkpoints"] = [r["update"] for r in read_jsonl(path)] == [312, 625, 937, 1250]
    result = dict(missing=missing, checks=checks,
                  status="complete_artifact_set" if not missing and all(checks.values()) else "incomplete",
                  note="Human interpretation and report writing remain your responsibility. Artifact presence alone does not verify results.")
    write_json(root / "status.json", result)
    print(json.dumps(result, indent=2))

"""Build tables/plots only from measured artifacts; audit missing experiments explicitly."""

import json
from pathlib import Path
import pandas as pd
from .common import *


def build(args=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = ROOT / "evidence"
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
    ax.set(xlabel="Optimizer-update attempt", ylabel="SFT response-token loss")
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
            ax.set_xlabel("Optimizer-update attempt")
        fig.tight_layout()
        fig.savefig(ev / "task1/regularization.png", dpi=160)
        plt.close(fig)
    results = []
    for path in (ev / "task1/evaluation").glob("*/scores.json"):
        scores = json.loads(path.read_text())
        for task, metrics in scores.items():
            for name, value in metrics.items():
                if isinstance(value, (float, int)):
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
    # Task 2 stage boundaries and image controls.
    for run in (ev / "task2").glob("tokens_*"):
        path = run / "train.jsonl"
        if path.exists():
            df = pd.read_json(path, lines=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df["update"], df.loss.rolling(25, min_periods=1).mean())
            for x in [2250, 3750]:
                ax.axvline(x, color="gray", linestyle="--")
            ax.set(xlabel="Main-run update", ylabel="Example-weighted loss")
            fig.tight_layout()
            fig.savefig(run / "learning_curve.png", dpi=160)
            plt.close(fig)
        path = run / "development.jsonl"
        if path.exists():
            df = pd.read_json(path, lines=True)
            table = df.pivot(
                index=["update", "suite"], columns="condition", values="mean_nll"
            ).reset_index()
            if "swap" in table:
                table["swap_penalty"] = table.swap - table.correct
            table.to_csv(run / "development_metrics.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            for (suite, condition), group in df.groupby(["suite", "condition"]):
                ax.plot(group["update"], group.mean_nll, label=f"{suite}/{condition}")
            for x in [2250, 3750]:
                ax.axvline(x, color="gray", linestyle="--")
            ax.set(xlabel="Main-run update", ylabel="Reference NLL / caption loss")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(run / "development_curves.png", dpi=160)
            plt.close(fig)
        path = run / "final/summary.json"
        if path.exists():
            pd.read_json(path).to_csv(run / "final/results.csv", index=False)
        # Generate actual paired-image sheets when image files are available at documented path.
        path = run / "final/qualitative_examples.json"
        if path.exists():
            from PIL import Image

            rows = json.loads(path.read_text())["records"]
            correct = [r for r in rows if r["condition"] == "correct"]
            for suite in ["ground", "natural"]:
                pairs = sorted({r["pair_id"] for r in correct if r["suite"] == suite})
                for pid in pairs:
                    pair = [r for r in correct if r["pair_id"] == pid]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    found = True
                    for ax, r in zip(axes, pair):
                        image = (
                            ROOT
                            / f'data/external/coco/images/{r["split"]}2014/COCO_{r["split"]}2014_{r["image_id"]:012d}.jpg'
                        )
                        if not image.exists():
                            found = False
                            break
                        ax.imshow(Image.open(image))
                        ax.axis("off")
                        ax.set_title(
                            f'Reference: {r["target"]}\nPrediction: {r["prediction"]}',
                            fontsize=9,
                        )
                    if found:
                        fig.suptitle(pair[0]["question"])
                        fig.tight_layout()
                        fig.savefig(run / f"final/{pid}.png", dpi=150)
                    plt.close(fig)
    # Task 3 accuracy/oracle/cost curves and signal trajectories.
    for run in (ev / "task3/offline").glob("*"):
        path = run / "fixed_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for method, g in df.groupby("method"):
            if method == "logp_rwc":
                continue
            axes[0].plot(g.N, g.accuracy, marker=".", label=method)
            axes[1].plot(g.token_cost, g.accuracy, marker=".", label=method)
        sc = df[df.method == "sc"]
        axes[0].plot(sc.N, sc.oracle, "k--", label="oracle coverage")
        axes[0].set(
            xlabel="Candidate budget N", ylabel="Answer accuracy / oracle coverage"
        )
        axes[1].set(xlabel="Consumed generated tokens (mean)", ylabel="Answer accuracy")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run / "accuracy_cost_curves.png", dpi=160)
        plt.close(fig)
        df[["method", "N", "sampling_failure", "selection_failure"]].to_csv(
            run / "failure_decomposition.csv", index=False
        )
        path = run / "illustrative_trajectories.json"
        if path.exists() and run.name == "mathvista":
            from .task3 import scores

            rows = json.loads(path.read_text())["records"]
            fig, axes = plt.subplots(
                len(rows), 1, figsize=(9, 2 * len(rows)), squeeze=False
            )
            for ax, r in zip(axes[:, 0], rows):
                k = np.asarray(r["kl_u_p"])
                v = np.asarray(r["image_attention"]) / (
                    np.asarray(r["image_attention"])
                    + r["prompt_attention"]
                    + r["generated_attention"]
                    + 1e-12
                )
                ax.plot(r["calibrated_confidence"], label="calibrated confidence")
                ax.plot(v, color="orange", alpha=0.7, label="visual engagement")
                ax.legend(fontsize=6)
                ax.set_title(f'{r["sample_id"]}, rollout {r["rollout_id"]}', fontsize=9)
                ax.set_ylabel("Confidence / image fraction")
            axes[-1, 0].set_xlabel("Generated token index")
            fig.tight_layout()
            fig.savefig(run / "signal_trajectories.png", dpi=150)
            plt.close(fig)
    audit()


def audit(args=None):
    required = {
        "task1": [
            "data.json",
            "learning_curves.png",
            "benchmark_comparison.csv",
            "primary_benchmarks.csv",
            "resources.csv",
            "diagnostics/metrics.json",
            "regularization.png",
            "displacements.csv",
            "qualitative/responses.json",
            "qualitative/annotations_to_complete.csv",
        ]
        + [
            f"{m}/resources.json"
            for m in ["full", "l2sp", "lora4", "lora16", "oplora16"]
        ]
        + [
            f"evaluation/{c}/scores.json"
            for c in [
                "baseline",
                "full_mid",
                "full",
                "l2sp",
                "lora4",
                "lora16",
                "oplora16",
            ]
        ],
        "task2": [
            "splits.json",
            "feature_cache.json",
            "tokens_196/completed.json",
            "tokens_196/calibration.json",
            "tokens_196/development_metrics.csv",
            "tokens_196/resources.jsonl",
            "tokens_196/final/predictions.json",
            "tokens_196/final/summary.json",
            "tokens_196/final/paired_contrasts.json",
            "tokens_196/final/qualitative_examples.json",
        ],
        "task3": [
            f"offline/{d}/{f}"
            for d in ["math", "mathvista"]
            for f in [
                "splits.json",
                "orders.json",
                "selected_policies.json",
                "fixed_summary.csv",
                "paired_differences.json",
                "signals.json",
                "adaptive_summary.json",
                "accuracy_cost_curves.png",
            ]
        ]
        + [
            "live/generations.json",
            "live/summary.csv",
            "live/revision_transitions.json",
        ],
    }
    status = {}
    for task, files in required.items():
        status[task] = {
            "present": [f for f in files if (ROOT / "evidence" / task / f).exists()],
            "missing": [
                f for f in files if not (ROOT / "evidence" / task / f).exists()
            ],
        }
    status["human_work"] = [
        "Complete 42 qualitative annotations, inspect labels, and write your own report/analysis."
    ]
    status["status"] = (
        "complete_artifact_set"
        if all(not status[t]["missing"] for t in required)
        else "experiments_incomplete"
    )
    status["note"] = (
        "Artifact presence is not proof of protocol compliance; inspect configs, skipped updates, counts, and raw records."
    )
    write_json(ROOT / "evidence/status.json", status)
    print(json.dumps({t: len(status[t]["missing"]) for t in required}, indent=2))

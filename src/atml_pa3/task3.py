from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from .answers import *
from .common import *

KEYS = ["dataset", "sample_id", "rollout_id"]
BUDGETS = [1, 2, 4, 8, 16, 32]


def scores(candidates):
    """Only revealed candidates enter percentile calibration."""

    def mean(x):
        return float(np.mean(x)) if len(x) else float("-inf")

    result = {
        "logp": np.array([mean(r["logp"]) for r in candidates]),
        "certainty": np.array([mean(r["kl_u_p"]) for r in candidates]),
        "shortest": -np.array([r["token_count"] for r in candidates], dtype=float),
    }
    if "image_attention" in candidates[0]:
        pooled = np.concatenate([r["kl_u_p"] for r in candidates])
        lo, hi = (
            np.quantile(pooled, [0.05, 0.95], method="linear")
            if len(pooled)
            else (0.0, 0.0)
        )
        vision = []
        confidence = []
        fused = []
        correlations = []
        for r in candidates:
            denom = (
                np.array(r["image_attention"])
                + r["prompt_attention"]
                + r["generated_attention"]
            )
            v = np.divide(
                r["image_attention"],
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom != 0,
            )
            c = np.clip((np.array(r["kl_u_p"]) - lo) / max(hi - lo, 1e-8), 0, 1)
            vision.append(float(v.mean()) if len(v) else 0.0)
            confidence.append(float(c.mean()) if len(c) else 0.0)
            fused.append(float((v * c).mean()) if len(c) else 0.0)
            correlations.append(
                float(np.corrcoef(c, v)[0, 1])
                if len(c) > 1 and np.std(c) > 0 and np.std(v) > 0
                else None
            )
        result.update(
            vision=np.array(vision),
            visconf=np.array(vision) * confidence,
            token_fusion=np.array(fused),
            confidence=np.array(confidence),
            confidence_vision_corr=correlations,
        )
    return result


def weighted_winner(groups, quality=None):
    n = len(groups)
    if quality is None or n == 1:
        weights = np.ones(n)
    else:
        weights = (n - rankdata(-np.asarray(quality), method="average")) / (n - 1)
    totals = {
        g: float(weights[np.asarray(groups) == g].sum()) for g in dict.fromkeys(groups)
    }
    # Dict insertion order is first supporting rollout, as required for ties.
    winner = max(totals, key=totals.get)
    ordered = sorted(totals.values(), reverse=True)
    margin = (
        (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / sum(ordered)
        if sum(ordered) > 0
        else 0.0
    )
    return int(winner), float(margin), totals


def select(groups, method, quality=None):
    if method == "first":
        return int(groups[0])
    if method == "sc":
        return weighted_winner(groups)[0]
    if method.endswith("_rwc"):
        return weighted_winner(groups, quality)[0]
    return int(groups[int(np.argmax(quality))])


def check_stop(candidates, tau, gamma=None, dataset="math"):
    groups = np.array([r["group"] for r in candidates])
    winner, _, votes = weighted_winner(groups)
    counts = sorted(votes.values(), reverse=True)
    representative = next(r["group_answer"] for r in candidates if r["group"] == winner)
    agreed = (
        representative != NO_ANSWER
        and (len(counts) == 1 or counts[0] > counts[1])
        and counts[0] / len(groups) >= tau
    )
    if not agreed:
        return False, winner
    if gamma is None:
        return True, winner
    quality = scores(candidates)["certainty" if dataset == "math" else "visconf"]
    rwc, margin, _ = weighted_winner(groups, quality)
    return rwc == winner and margin + 1e-12 >= gamma, winner


def simulate(candidates, tau=1.0, gamma=None, dataset="math", fallback=False):
    for n in [4, 8, 16]:
        if not fallback:
            stop, winner = check_stop(candidates[:n], tau, gamma, dataset)
            if stop:
                return n, winner
    return 32, weighted_winner([r["group"] for r in candidates])[0]


DEFAULT_SCHEMA = {
    "generations": {
        "token_ids": "token_ids",
        "token_count": "token_count",
        "extracted_answer": "extracted_answer",
        "gold_answer": "gold_answer",
        "problem": "problem",
    },
    "probability": {"logp": "logp", "kl_u_p": "kl_u_p"},
    "attention": {
        "image_attention": "image_attention",
        "prompt_attention": "prompt_attention",
        "generated_attention": "generated_attention",
    },
}


def load_tables(folder, dataset, schema, metadata):
    folder = Path(folder)
    tables = []
    for kind, file in [
        ("generations", "generations.parquet"),
        ("probability", "probability_metrics.parquet"),
    ] + (
        [("attention", "attention_metrics.parquet")] if dataset == "mathvista" else []
    ):
        frame = pd.read_parquet(folder / file)
        mapping = schema[kind]
        frame = frame.rename(columns={v: k for k, v in mapping.items()})
        if kind == "generations" and dataset == "mathvista" and "problem" not in frame:
            frame["problem"] = [
                metadata.get(str(q), {}).get("question", "") for q in frame["sample_id"]
            ]
        required = KEYS + list(mapping)
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(
                f"{file} missing {missing}; adapt configs/rollout_schema.json from supplied README.MD"
            )
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate join keys in {file}")
        tables.append(frame[required])
    joined = tables[0]
    for table in tables[1:]:
        result = joined.merge(
            table, on=KEYS, how="outer", validate="one_to_one", indicator=True
        )
        if (result["_merge"] != "both").any():
            raise ValueError("Unmatched rollout keys across tables")
        joined = result.drop(columns="_merge")
    pools = {}
    all_ids = sorted(joined.sample_id.astype(str).unique())
    if len(all_ids) != 1000:
        raise ValueError(f"{dataset}: expected 1000 questions, got {len(all_ids)}")
    for qid, group in joined.assign(sample_id=joined.sample_id.astype(str)).groupby(
        "sample_id", sort=True
    ):
        if len(group) != 32:
            raise ValueError(f"{qid}: expected 32 rollouts")
        rows = group.sort_values("rollout_id").to_dict("records")
        meta = metadata.get(qid, {}) if dataset == "mathvista" else None
        if dataset == "mathvista" and not meta:
            raise ValueError(f"Missing MathVista metadata for {qid}")
        for r in rows:
            if int(r["token_count"]) != r["token_count"] or r["token_count"] < 0:
                raise ValueError("Retained token count must be a nonnegative integer")
            if r["gold_answer"] is None or pd.isna(r["gold_answer"]):
                raise ValueError("Missing reference answer in supplied data")
            for key in ["token_ids", "logp", "kl_u_p"] + (
                ["image_attention", "prompt_attention", "generated_attention"]
                if dataset == "mathvista"
                else []
            ):
                r[key] = np.asarray(r[key])
                if r[key].ndim != 1:
                    raise ValueError(f"Expected a one-dimensional token list: {key}")
                if len(r[key]) != int(r["token_count"]):
                    raise ValueError(f"Token alignment failure: {qid}, {key}")
            if not np.isfinite(r["logp"]).all() or not np.isfinite(r["kl_u_p"]).all():
                raise ValueError("Non-finite probability values")
            if dataset == "mathvista":
                for key in [
                    "image_attention",
                    "prompt_attention",
                    "generated_attention",
                ]:
                    if not np.isfinite(r[key]).all() or np.any(np.asarray(r[key]) < 0):
                        raise ValueError("Invalid attention totals")
                if np.any(
                    np.asarray(r["image_attention"])
                    + r["prompt_attention"]
                    + r["generated_attention"]
                    <= 0
                ):
                    raise ValueError(
                        "Zero total attention mass; check supplied field mapping"
                    )
            if pd.isna(r["extracted_answer"]):
                r["extracted_answer"] = None
        groups, reps = answer_groups(
            [r["extracted_answer"] for r in rows], dataset, meta
        )
        for r, g in zip(rows, groups):
            r["group"] = int(g)
            r["group_answer"] = reps[int(g)]
            r["correct"] = (
                math_equal(r["extracted_answer"], r["gold_answer"])
                if dataset == "math"
                else (
                    mathvista_normalize(r["extracted_answer"], meta) != NO_ANSWER
                    and mathvista_normalize(r["extracted_answer"], meta)
                    == mathvista_normalize(r["gold_answer"], meta)
                )
            )
        # Never use gold to construct equivalence classes or resolve a vote tie.
        pools[str(qid)] = rows
    return pools


def make_splits(pools):
    ids = sorted(pools)
    ids = np.array(ids)[np.random.default_rng(SEED).permutation(len(ids))].tolist()
    return dict(dev=ids[:200], evaluation=ids[200:], live=ids[200:250])


def question_orders(dataset, qid):
    stable = int.from_bytes(
        hashlib.sha256(f"{SEED}:{dataset}:{qid}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(stable)
    return [rng.permutation(32).tolist() for _ in range(10)]


def is_correct(rows, group):
    return bool(next(r["correct"] for r in rows if r["group"] == group))


def evaluate_fixed(pools, ids, dataset, orders):
    records = []
    signals = []
    trajectories = []
    for qid in ids:
        original = pools[qid]
        for order_id, order in enumerate(orders[qid]):
            rows = [original[i] for i in order]
            for n in BUDGETS:
                current = rows[:n]
                s = scores(current)
                groups = np.array([r["group"] for r in current])
                methods = {
                    "first": None,
                    "sc": None,
                    "logp": s["logp"],
                    "logp_rwc": s["logp"],
                    "certainty": s["certainty"],
                    "certainty_rwc": s["certainty"],
                    "shortest": s["shortest"],
                }
                if dataset == "mathvista":
                    for score in ["vision", "visconf", "token_fusion"]:
                        methods[score] = s[score]
                        methods[score + "_rwc"] = s[score]
                oracle = any(r["correct"] for r in current)
                for method, quality in methods.items():
                    winner = select(groups, method, quality)
                    correct = is_correct(current, winner)
                    consumed = current[:1] if method == "first" else current
                    selected = (
                        current[int(np.argmax(quality))]
                        if quality is not None and not method.endswith("_rwc")
                        else next(r for r in current if r["group"] == winner)
                    )
                    records.append(
                        dict(
                            dataset=dataset,
                            sample_id=qid,
                            order=order_id,
                            N=n,
                            method=method,
                            accuracy=float(correct),
                            oracle=float(oracle),
                            distinct=len(set(groups)),
                            sampling_failure=float(not oracle),
                            selection_failure=float(oracle and not correct),
                            token_cost=sum(r["token_count"] for r in consumed),
                            rollouts=len(consumed),
                            selected_group=winner,
                            selected_answer=selected["group_answer"],
                            selected_extracted_answer=selected["extracted_answer"],
                            selected_rollout_id=selected["rollout_id"],
                            candidate_ids=[r["rollout_id"] for r in current],
                        )
                    )
        # Signal diagnostics use each question once at N=32, not ten identical pools.
        s = scores(original)
        y = np.array([r["correct"] for r in original], float)
        for score in ["logp", "certainty", "shortest"] + (
            ["vision", "visconf", "token_fusion"] if dataset == "mathvista" else []
        ):
            values = s[score]
            idx = np.argsort(values, kind="stable")
            corr = (
                float(spearmanr(values, [r["token_count"] for r in original]).statistic)
                if np.std(values) > 0
                and np.std([r["token_count"] for r in original]) > 0
                else None
            )
            signals.append(
                dict(
                    sample_id=qid,
                    score=score,
                    length_spearman=corr,
                    correctness_spearman=(
                        float(spearmanr(values, y).statistic)
                        if np.std(values) > 0 and np.std(y) > 0
                        else None
                    ),
                    top_quartile_accuracy=float(y[idx[-8:]].mean()),
                    bottom_quartile_accuracy=float(y[idx[:8]].mean()),
                    top_minus_bottom=float(y[idx[-8:]].mean() - y[idx[:8]].mean()),
                )
            )
        if dataset == "mathvista":
            corrs = [x for x in s["confidence_vision_corr"] if x is not None]
            signals.append(
                dict(
                    sample_id=qid,
                    score="confidence_vision",
                    correlation=float(np.mean(corrs)) if corrs else None,
                    valid_rollouts=len(corrs),
                    fusion_covariance=float(np.mean(s["token_fusion"] - s["visconf"])),
                )
            )
    return pd.DataFrame(records), signals


def stopping_states(rows, dataset):
    states = {}
    for n in [4, 8, 16, 32]:
        current = rows[:n]
        groups = [r["group"] for r in current]
        winner, _, votes = weighted_winner(groups)
        ordered = sorted(votes.values(), reverse=True)
        quality = scores(current)["certainty" if dataset == "math" else "visconf"]
        rwc, margin, _ = weighted_winner(groups, quality)
        states[n] = dict(
            winner=winner,
            rwc=rwc,
            margin=margin,
            nonempty=next(r["group_answer"] for r in current if r["group"] == winner)
            != NO_ANSWER,
            unique=len(ordered) == 1 or ordered[0] > ordered[1],
            fraction=ordered[0] / n,
        )
    return states


def policy_results(pools, ids, dataset, orders, tau, gamma, fallback=False, cache=None):
    result = []
    for qid in ids:
        for oi, order in enumerate(orders[qid]):
            rows = [pools[qid][i] for i in order]
            if cache is None:
                n, winner = simulate(rows, tau, gamma, dataset, fallback)
            else:
                states = cache[(qid, oi)]
                n = 32
                if not fallback:
                    for check in [4, 8, 16]:
                        st = states[check]
                        if (
                            st["unique"]
                            and st["nonempty"]
                            and st["fraction"] >= tau
                            and (
                                gamma is None
                                or (
                                    st["rwc"] == st["winner"]
                                    and st["margin"] + 1e-12 >= gamma
                                )
                            )
                        ):
                            n = check
                            break
                winner = states[n]["winner"]
            correct = is_correct(rows[:n], winner)
            fixed = weighted_winner([r["group"] for r in rows])[0]
            result.append(
                dict(
                    sample_id=qid,
                    order=oi,
                    accuracy=float(correct),
                    token_cost=sum(r["token_count"] for r in rows[:n]),
                    rollouts=n,
                    selected_group=winner,
                    selected_answer=next(
                        r["group_answer"] for r in rows[:n] if r["group"] == winner
                    ),
                    candidate_ids=[r["rollout_id"] for r in rows[:n]],
                    wrong_early_stop=n < 32 and not correct,
                    recoverable_by_32=n < 32
                    and not correct
                    and is_correct(rows, fixed),
                    full_pool_correct_fraction=float(
                        np.mean([r["correct"] for r in rows])
                    ),
                )
            )
    return pd.DataFrame(result)


def tune(pools, ids, dataset, orders):
    cache = {
        (qid, oi): stopping_states([pools[qid][i] for i in order], dataset)
        for qid in ids
        for oi, order in enumerate(orders[qid])
    }
    baseline = policy_results(pools, ids, dataset, orders, 1, None, True, cache)
    target = float(baseline.accuracy.mean()) - 0.01
    rows = []
    selected = {}
    for family in ["agreement", "quality"]:
        candidates = []
        for tau in [0.5, 0.625, 0.75, 0.875, 1.0]:
            for gamma in [None] if family == "agreement" else [0.0, 0.1, 0.25, 0.5]:
                values = policy_results(
                    pools, ids, dataset, orders, tau, gamma, cache=cache
                )
                candidates.append(
                    dict(
                        family=family,
                        tau=tau,
                        gamma=gamma,
                        fallback=False,
                        accuracy=float(values.accuracy.mean()),
                        token_cost=float(values.token_cost.mean()),
                    )
                )
        candidates.append(
            dict(
                family=family,
                tau=1.0,
                gamma=None if family == "agreement" else 0.0,
                fallback=True,
                accuracy=float(baseline.accuracy.mean()),
                token_cost=float(baseline.token_cost.mean()),
            )
        )
        eligible = [p for p in candidates if p["accuracy"] + 1e-12 >= target]
        chosen = min(
            eligible,
            key=lambda p: (
                p["token_cost"],
                -p["accuracy"],
                p["tau"],
                p["gamma"] or 0,
                p["fallback"],
            ),
        )
        selected[family] = chosen
        rows.extend(candidates)
    return selected, rows


def run(args):
    seed_all()
    import importlib.metadata

    schema = json.loads(Path(args.schema).read_text())
    meta = json.loads(Path(args.metadata).read_text())
    if isinstance(meta, list):
        meta = {str(r.get("pid", r.get("sample_id"))): r for r in meta}
    out = ROOT / "evidence/task3/offline"
    out.mkdir(parents=True, exist_ok=True)
    for dataset, folder in [("math", args.math_dir), ("mathvista", args.mathvista_dir)]:
        if not (Path(folder) / "README.MD").exists() and not args.readme:
            raise FileNotFoundError(
                "Supply --readme with the original rollout README.MD for provenance"
            )
        pools = load_tables(folder, dataset, schema, meta)
        splits = make_splits(pools)
        orders = {q: question_orders(dataset, q) for q in pools}
        dest = out / dataset
        dest.mkdir(exist_ok=True)
        write_json(dest / "splits.json", splits)
        write_json(
            dest / "orders.json",
            {
                q: [[pools[q][i]["rollout_id"] for i in order] for order in orders[q]]
                for q in pools
            },
        )
        selected, grid = tune(pools, splits["dev"], dataset, orders)
        write_json(dest / "selected_policies.json", selected)
        write_json(dest / "development_grid.json", grid)
        # Freeze dev choices on disk before computing held-out results.
        fixed, signals = evaluate_fixed(pools, splits["evaluation"], dataset, orders)
        fixed.to_parquet(dest / "per_question_fixed.parquet", index=False)
        write_json(dest / "signals.json", signals)
        avg = (
            fixed.groupby(["sample_id", "method", "N"])[
                [
                    "accuracy",
                    "oracle",
                    "token_cost",
                    "rollouts",
                    "distinct",
                    "sampling_failure",
                    "selection_failure",
                ]
            ]
            .mean()
            .reset_index()
        )
        summary = avg.groupby(["method", "N"]).mean(numeric_only=True).reset_index()
        summary.to_csv(dest / "fixed_summary.csv", index=False)
        sc = avg[avg.method == "sc"].pivot(
            index="sample_id", columns="N", values="accuracy"
        )
        contrasts = []
        for (method, n), group in avg.groupby(["method", "N"]):
            delta = (
                group.set_index("sample_id").accuracy.sort_index() - sc[n].sort_index()
            )
            contrasts.append(
                dict(
                    method=method,
                    N=int(n),
                    reference="self-consistency at same N",
                    difference=float(delta.mean()),
                    ci=pair_bootstrap(delta, 1000),
                )
            )
        write_json(dest / "paired_differences.json", contrasts)
        adaptive = []
        for family, policy in selected.items():
            result = policy_results(
                pools,
                splits["evaluation"],
                dataset,
                orders,
                policy["tau"],
                policy["gamma"],
                policy["fallback"],
            )
            result.to_parquet(dest / f"adaptive_{family}.parquet", index=False)
            byq = result.groupby("sample_id").mean(numeric_only=True)
            delta = byq.accuracy.sort_index() - sc[32].sort_index()
            adaptive.append(
                dict(
                    family=family,
                    **policy,
                    heldout_accuracy=float(byq.accuracy.mean()),
                    heldout_token_cost=float(byq.token_cost.mean()),
                    mean_rollouts=float(byq.rollouts.mean()),
                    stopping_distribution={
                        str(k): float(v)
                        for k, v in result.rollouts.value_counts(normalize=True).items()
                    },
                    accuracy_difference_vs_fixed32=float(delta.mean()),
                    ci=pair_bootstrap(delta, 1000),
                )
            )
            examples = (
                result[result.recoverable_by_32]
                .sort_values(["sample_id", "order"])
                .head(3)
                .to_dict("records")
            )
            write_json(
                dest / f"premature_stop_examples_{family}.json",
                dict(
                    selection="first three recoverable early stops sorted by question ID/order",
                    records=examples,
                ),
            )
        write_json(dest / "adaptive_summary.json", adaptive)
        # Preselected trajectories are independent of scores and labels.
        trajectories = []
        for qid in sorted(splits["evaluation"])[:3]:
            pool = pools[qid]
            if dataset == "mathvista":
                lo, hi = np.quantile(
                    np.concatenate([r["kl_u_p"] for r in pool]),
                    [0.05, 0.95],
                    method="linear",
                )
                for r in pool[:2]:
                    denom = (
                        np.asarray(r["image_attention"])
                        + r["prompt_attention"]
                        + r["generated_attention"]
                    )
                    v = np.divide(
                        r["image_attention"],
                        denom,
                        out=np.zeros_like(denom, dtype=float),
                        where=denom != 0,
                    )
                    trajectories.append(
                        dict(
                            r,
                            calibrated_confidence=np.clip(
                                (np.asarray(r["kl_u_p"]) - lo) / max(hi - lo, 1e-8),
                                0,
                                1,
                            ),
                            visual_engagement=v,
                        )
                    )
            else:
                trajectories.extend(pool[:2])
        write_json(
            dest / "illustrative_trajectories.json",
            dict(
                selection="first three held-out IDs lexicographically; first two stored rollout IDs",
                records=trajectories,
            ),
        )
        disagreements = fixed[
            (fixed.N == 32)
            & (fixed.method.isin(["sc", "certainty", "certainty_rwc", "visconf"]))
        ]
        ids = disagreements.groupby(["sample_id", "order"]).selected_group.nunique()
        chosen = list(ids[ids > 1].sort_index().head(3).index)
        aligned = [
            r
            for r in disagreements.to_dict("records")
            if (r["sample_id"], r["order"]) in chosen
        ]
        write_json(
            dest / "selection_disagreements.json",
            dict(
                selection="first three question/order pairs with selector disagreement",
                records=aligned,
            ),
        )
        files = [p for p in Path(folder).glob("*.parquet")]
        readme = Path(args.readme) if args.readme else Path(folder) / "README.MD"
        write_json(
            dest / "provenance.json",
            dict(
                seed=SEED,
                math_verify=importlib.metadata.version("math-verify"),
                schema=schema,
                files={p.name: sha256(p) for p in files},
                readme_sha256=sha256(readme),
                scoring="math-verify equivalence; MathVista choices/answer_type/precision",
                full_questions=len(pools),
            ),
        )
        # Save canonical live inputs, never synthesize original problems or revisions.
        if dataset == "math":
            write_json(
                ROOT / "data/prepared/task3/live_questions.json",
                [
                    dict(
                        sample_id=qid,
                        problem=pools[qid][0]["problem"],
                        gold_answer=pools[qid][0]["gold_answer"],
                    )
                    for qid in splits["live"]
                ],
            )
        print(dataset, "offline evaluation complete", flush=True)

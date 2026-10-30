"""Deterministic COCO/VQAv2 split construction and pair-preserving sampling."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import urllib.request
import numpy as np
import torch
from .answers import vqa_normalize
from .common import *


def read(path):
    return json.loads(Path(path).read_text())


def natural_pool(root, split):
    questions = read(root / f"v2_OpenEnded_mscoco_{split}2014_questions.json")[
        "questions"
    ]
    annotations = read(root / f"v2_mscoco_{split}2014_annotations.json")["annotations"]
    qs = {r["question_id"]: r for r in questions}
    anns = {r["question_id"]: r for r in annotations}
    pairs = read(root / f"v2_mscoco_{split}2014_complementary_pairs.json")
    if isinstance(pairs, dict):
        pairs = pairs.get("complementary_pairs", pairs.get("pairs"))
    if not isinstance(pairs, list):
        raise ValueError(
            "Expected complementary pair list of [question_id,question_id]"
        )
    pool = []
    for pair in pairs:
        if isinstance(pair, dict):
            pair = [pair["question_id"], pair["complementary_question_id"]]
        a, b = map(int, pair)
        rows = []
        for qid in [a, b]:
            an = anns[qid]
            counts = Counter(vqa_normalize(x["answer"]) for x in an["answers"])
            canonical, count = counts.most_common(1)[0]
            if count < 8:
                break
            target = next(
                x["answer"]
                for x in an["answers"]
                if vqa_normalize(x["answer"]) == canonical
            )
            rows.append(
                dict(
                    question_id=qid,
                    image_id=an["image_id"],
                    question=qs[qid]["question"],
                    target=target,
                    answers=[x["answer"] for x in an["answers"]],
                    canonical=canonical,
                    kind=an["answer_type"],
                    split=split,
                )
            )
        if (
            len(rows) == 2
            and rows[0]["canonical"] != rows[1]["canonical"]
            and rows[0]["question"] == rows[1]["question"]
            and rows[0]["image_id"] != rows[1]["image_id"]
        ):
            pool.append(rows)
    return pool


def select_natural(pool, quota, rng, blocked):
    selected = []
    seen = set()
    counts = Counter()
    for i in rng.permutation(len(pool)):
        pair = pool[int(i)]
        kind = pair[0]["kind"]
        ids = {r["image_id"] for r in pair}
        signature = tuple(sorted(r["question_id"] for r in pair))
        if (
            signature in seen
            or ids & blocked
            or kind != pair[1]["kind"]
            or counts[kind] >= quota.get(kind, 0)
        ):
            continue
        # Repeated images within a split are allowed; between splits they are forbidden.
        seen.add(signature)
        selected.append(pair)
        counts[kind] += 1
        if all(counts[k] == v for k, v in quota.items()):
            break
    if any(counts[k] != v for k, v in quota.items()):
        raise ValueError(
            f"Insufficient image-disjoint natural pairs: {counts}, need {quota}"
        )
    blocked.update(r["image_id"] for p in selected for r in p)
    return selected


def ground_pool(instances, available, count, rng, split):
    categories = {r["id"]: r["name"] for r in instances["categories"]}
    by = defaultdict(Counter)
    for r in instances["annotations"]:
        by[r["image_id"]][r["category_id"]] += 1
    available = sorted(available)
    pool = []
    # Alternate presence/counting; counts based on all annotated instances, including crowd entries.
    for cid in rng.permutation(sorted(categories)):
        name = categories[int(cid)]
        neg = [i for i in available if by[i][int(cid)] == 0]
        pos = [i for i in available if by[i][int(cid)] > 0]
        buckets = {
            n: [i for i in available if by[i][int(cid)] == n] for n in range(1, 5)
        }
        for kind in ["presence", "counting"]:
            if kind == "presence":
                if not neg or not pos:
                    continue
                combos = [
                    (int(a), int(b), "yes", "no")
                    for a, b in zip(
                        rng.permutation(pos), rng.choice(neg, len(pos), replace=True)
                    )
                ]
            else:
                combos = []
                for n in range(1, 5):
                    for m in range(n + 1, 5):
                        if buckets[n] and buckets[m]:
                            combos.extend(
                                (int(a), int(b), str(n), str(m))
                                for a, b in zip(
                                    rng.permutation(buckets[n]),
                                    rng.choice(
                                        buckets[m], len(buckets[n]), replace=True
                                    ),
                                )
                            )
            rng.shuffle(combos)
            # Each category contributes a bounded reusable pool; final balancing is below.
            for a, b, ya, yb in combos[: max(4, count // len(categories) + 2)]:
                q = (
                    f"Does this image contain {name}?"
                    if kind == "presence"
                    else f"How many {name} can you see?"
                )
                pool.append(
                    [
                        dict(
                            image_id=i,
                            question=q,
                            target=y,
                            canonical=y,
                            answers=[y],
                            kind=kind,
                            split=split,
                        )
                        for i, y in [(a, ya), (b, yb)]
                    ]
                )
    final = []
    for kind in ["presence", "counting"]:
        choices = [p for p in pool if p[0]["kind"] == kind]
        rng.shuffle(choices)
        if len(choices) < count // 2:
            raise ValueError(f"Insufficient {kind} grounding pool")
        final.extend(choices[: count // 2])
    rng.shuffle(final)
    return final


def flatten(pairs, prefix):
    records = []
    for i, pair in enumerate(pairs):
        for side, r in enumerate(pair):
            records.append(
                dict(
                    r,
                    pair_id=f"{prefix}_{i:06}",
                    id=f"{prefix}_{i:06}_{side}",
                    partner=f"{prefix}_{i:06}_{1-side}",
                )
            )
    return records


def prepare(args):
    root = Path(args.annotations)
    out = ROOT / "data/prepared/task2"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train_inst = read(root / "instances_train2014.json")
    val_inst = read(root / "instances_val2014.json")
    train_caps = read(root / "captions_train2014.json")
    val_caps = read(root / "captions_val2014.json")
    all_train = {r["id"] for r in train_inst["images"]}
    all_val = {r["id"] for r in val_inst["images"]}
    used = set()
    val_pool = natural_pool(root, "val")
    nat_dev = select_natural(
        val_pool, {"yes/no": 12, "number": 8, "other": 12}, rng, used
    )
    nat_test = select_natural(
        val_pool, {"yes/no": 36, "number": 24, "other": 36}, rng, used
    )
    # Reserve disjoint images for grounding dev/test before building their pairs.
    remaining = rng.permutation(sorted(all_val - used))
    dev_images = set(map(int, remaining[:2000]))
    test_images = set(map(int, remaining[2000:8000]))
    gr_dev = ground_pool(val_inst, dev_images, 24, rng, "val")
    gr_test = ground_pool(val_inst, test_images, 64, rng, "val")
    used.update(r["image_id"] for p in gr_dev + gr_test for r in p)
    nat_train = select_natural(
        natural_pool(root, "train"),
        {"yes/no": 640, "number": 256, "other": 640},
        rng,
        set(),
    )
    gr_train = ground_pool(train_inst, all_train, 4096, rng, "train")
    cap_maps = []
    for caps, ids, num, per, split in [
        (train_caps, all_train, 6000, 3, "train"),
        (val_caps, all_val - used, 96, 1, "val"),
    ]:
        by = defaultdict(list)
        for r in caps["annotations"]:
            by[r["image_id"]].append(r)
        eligible = sorted(i for i in ids if len(by[i]) >= per)
        if len(eligible) < num:
            raise ValueError("Insufficient captions")
        chosen = rng.choice(eligible, num, replace=False)
        records = []
        for i in chosen:
            for r in sorted(by[int(i)], key=lambda x: x["id"])[:per]:
                records.append(
                    dict(
                        id=f'caption_{r["id"]}',
                        image_id=int(i),
                        question="Describe this image.",
                        target=r["caption"],
                        kind="caption",
                        split=split,
                        caption_id=r["id"],
                    )
                )
        cap_maps.append(records)
    train = (
        flatten(nat_train, "natural_train")
        + flatten(gr_train, "ground_train")
        + cap_maps[0]
    )
    dev = flatten(nat_dev, "natural_dev") + flatten(gr_dev, "ground_dev") + cap_maps[1]
    test = flatten(nat_test, "natural_test") + flatten(gr_test, "ground_test")
    assert not ({r["image_id"] for r in dev} & {r["image_id"] for r in test})
    # Numeric image IDs alone do not determine a COCO split; record split-qualified identities.
    assert not (
        {(r["split"], r["image_id"]) for r in train}
        & {(r["split"], r["image_id"]) for r in dev + test}
    )
    for suite in ["natural", "ground"]:
        rows = [r for r in test if r["id"].startswith(suite)]
        pairs = sorted({r["pair_id"] for r in rows})
        for r in rows:
            candidates = [
                x
                for x in rows
                if x["pair_id"] != r["pair_id"]
                and x["image_id"]
                not in {
                    r["image_id"],
                    next(y["image_id"] for y in rows if y["id"] == r["partner"]),
                }
            ]
            if not candidates:
                raise ValueError("No valid mismatch mapping")
            r["mismatched_image_id"] = candidates[int(rng.integers(len(candidates)))][
                "image_id"
            ]
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        write_json(out / f"{name}.json", rows)
    write_json(
        ROOT / "evidence/task2/splits.json",
        dict(
            train=train,
            dev=dev,
            test=test,
            seed=SEED,
            annotations_sha256={p.name: sha256(p) for p in sorted(root.glob("*.json"))},
        ),
    )
    print(
        "Saved split/example/pair IDs, all references, and fixed mismatched-image mapping."
    )


def cache(args):
    from transformers import AutoImageProcessor, SiglipVisionModel
    from huggingface_hub import model_info
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    require_cuda()
    seed_all()
    out = ROOT / "data/cache/task2"
    out.mkdir(parents=True, exist_ok=True)
    records = sum(
        [
            read(ROOT / f"data/prepared/task2/{s}.json")
            for s in ["train", "dev", "test"]
        ],
        [],
    )
    ids = sorted({(r["split"], r["image_id"]) for r in records})
    # Optional downloads fetch only selected images, never the entire COCO archive.
    images = Path(args.images)

    def fetch(item):
        split, i = item
        path = images / f"{split}2014" / f"COCO_{split}2014_{i:012d}.jpg"
        if not path.exists():
            if not args.download:
                raise FileNotFoundError(f"{path}; use --download or supply COCO images")
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"https://images.cocodataset.org/{split}2014/{path.name}",
                str(path) + ".tmp",
            )
            Path(str(path) + ".tmp").replace(path)
        return path

    paths = list(ThreadPoolExecutor(max_workers=8).map(fetch, ids))
    name = "google/siglip-base-patch16-224"
    manifest = ROOT / "evidence/task2/feature_cache.json"
    revision = read(manifest)["revision"] if manifest.exists() else model_info(name).sha
    processor = AutoImageProcessor.from_pretrained(name, revision=revision)
    encoder = (
        SiglipVisionModel.from_pretrained(
            name, revision=revision, torch_dtype=torch.float16
        )
        .to("cuda")
        .eval()
    )
    encoder.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for j in range(0, len(ids), 64):
        chunk = ids[j : j + 64]
        destinations = [out / f"{s}_{i}.pt" for s, i in chunk]
        if all(p.exists() for p in destinations):
            continue
        imgs = [Image.open(p).convert("RGB") for p in paths[j : j + 64]]
        inputs = processor(images=imgs, return_tensors="pt").to(
            "cuda", dtype=torch.float16
        )
        with torch.no_grad():
            features = encoder(**inputs).last_hidden_state
        # SiglipVisionTransformer.forward already applies post_layernorm to last_hidden_state.
        if tuple(features.shape[1:]) != (196, 768):
            raise ValueError(features.shape)
        for path, feature in zip(destinations, features):
            torch.save(feature.cpu().half(), path)
    torch.cuda.synchronize()
    write_json(
        ROOT / "evidence/task2/feature_cache.json",
        dict(
            images=len(ids),
            seconds=time.perf_counter() - start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            model=name,
            revision=revision,
            normalization="SiglipVisionModel.last_hidden_state; post_layernorm applied exactly once",
            dtype="fp16",
        ),
    )
    del encoder
    torch.cuda.empty_cache()

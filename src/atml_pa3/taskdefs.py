"""Stock harness tasks, with immutable dataset revisions and pinned task code."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .common import ARTIFACT_ROOT, write_json

TASKS = {
    "gsm8k_cot": ("openai/gsm8k", "main", 1319),
    "hellaswag": ("Rowan/hellaswag", None, 10042),
    "boolq": ("aps/super_glue", "boolq", 3270),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", 2376),
}


def definition_hash():
    import lm_eval
    root = Path(lm_eval.__file__).parent
    h = hashlib.sha256()
    # Pin all task YAML/Python sources, including any stock includes and utilities.
    for p in sorted((root / "tasks").rglob("*")):
        if p.suffix in (".yaml", ".py"):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def dataset_spec(create=False):
    path = ARTIFACT_ROOT / "data/prepared/datasets.json"
    if path.exists():
        spec = json.loads(path.read_text())
        if spec["task_definition_sha256"] != definition_hash():
            raise ValueError("Harness task definitions changed; use the pinned environment")
        return spec
    if not create:
        raise FileNotFoundError("Run pa3 task1 prepare first")
    from huggingface_hub import HfApi
    api = HfApi()
    names = {value[0] for value in TASKS.values()} | {"meta-math/MetaMathQA"}
    spec = dict(revisions={name: api.dataset_info(name).sha for name in sorted(names)},
                lm_eval=importlib.metadata.version("lm_eval"), task_definition_sha256=definition_hash())
    write_json(path, spec)
    return spec


def task_configs(names):
    spec = dataset_spec()
    return [dict(task=name, dataset_path=TASKS[name][0], dataset_name=TASKS[name][1],
                 dataset_kwargs=dict(revision=spec["revisions"][TASKS[name][0]], trust_remote_code=False))
            for name in names]


def load_tasks(names):
    from lm_eval.tasks import TaskManager, get_task_dict
    return get_task_dict(task_configs(names), TaskManager())

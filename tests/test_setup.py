"""Regressions for the reported Colab shell and torchvision startup failures."""

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


def installer_module():
    spec = importlib.util.spec_from_file_location(
        "setup_colab", ROOT / "scripts/setup_colab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_does_not_overwrite_an_existing_directory(tmp_path):
    directory = tmp_path / "important"
    directory.mkdir()
    (directory / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a virtual environment"):
        installer_module().validate_existing(directory, False)
    assert (directory / "notes.txt").read_text() == "keep"


def test_shared_site_packages_are_rejected(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="isolated"):
        installer_module().validate_existing(tmp_path, False)


def test_cpu_cuda_profiles_cannot_be_silently_mixed(tmp_path, monkeypatch):
    module = installer_module()
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (tmp_path / "pa3_setup.json").write_text(json.dumps({"cpu": True}))
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **k: f"{sys.version_info.major}.{sys.version_info.minor}\n",
    )
    with pytest.raises(ValueError, match="profile differs"):
        module.validate_existing(tmp_path, False)


def test_notebook_shells_activate_venv_and_use_bash_syntax():
    notebook = json.loads((ROOT / "notebooks/ATML_PA3_Colab.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if not source.startswith("%%bash\n"):
            continue
        body = source.split("\n", 1)[1]
        assert "source .venv-colab/bin/activate" in body
        assert not re.search(
            r"^\s*!\S", body, re.MULTILINE
        ), "IPython !command is not a bash command"
        subprocess.run(["bash", "-n"], input=body, text=True, check=True)
    setup = "".join(notebook["cells"][2]["source"])
    assert "scripts/setup_colab.py" in setup and "os.environ['PATH']" in setup


def test_doctor_reports_missing_cuda_without_a_task_import(monkeypatch, capsys):
    import torch
    from atml_pa3.cli import main

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "atml_pa3.task3", None)
    monkeypatch.setattr(sys, "argv", ["pa3", "doctor", "--require-cuda"])
    with pytest.raises(SystemExit, match="Setup failed"):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["checks"]["torchvision_operator"]
    assert any("CUDA is unavailable" in error for error in result["errors"])


def test_doctor_catches_the_reported_missing_torchvision_operator(monkeypatch, capsys):
    import torchvision
    from atml_pa3.doctor import doctor

    def missing_operator(*args, **kwargs):
        raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(torchvision.ops, "nms", missing_operator)
    with pytest.raises(SystemExit, match="Setup failed"):
        doctor(SimpleNamespace(require_cuda=False))
    result = json.loads(capsys.readouterr().out)
    assert any("torchvision::nms" in error for error in result["errors"])

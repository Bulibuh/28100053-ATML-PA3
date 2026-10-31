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


def test_verified_pip_download_rejects_bad_checksum_without_executing(
    tmp_path, monkeypatch
):
    import hashlib
    import io

    module = installer_module()
    wheel = b"actual wheel bytes"
    metadata = {
        "urls": [
            {
                "filename": f"pip-{module.PIP_VERSION}-py3-none-any.whl",
                "url": "https://files.pythonhosted.org/example.whl",
                "digests": {"sha256": hashlib.sha256(b"different bytes").hexdigest()},
            }
        ]
    }
    responses = iter([io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(wheel)])
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: next(responses)
    )
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        module.download_verified_pip(tmp_path)
    assert not list(tmp_path.rglob("*.whl"))


def test_verified_pip_download_retains_exact_wheel(tmp_path, monkeypatch):
    import hashlib
    import io

    module = installer_module()
    wheel = b"verified wheel contents"
    metadata = {
        "urls": [
            {
                "filename": f"pip-{module.PIP_VERSION}-py3-none-any.whl",
                "url": "https://files.pythonhosted.org/example.whl",
                "digests": {"sha256": hashlib.sha256(wheel).hexdigest()},
            }
        ]
    }
    responses = iter([io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(wheel)])
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: next(responses)
    )
    path = module.download_verified_pip(tmp_path)
    assert path.read_bytes() == wheel


def test_pipless_partial_venv_is_repaired_without_ensurepip(tmp_path, monkeypatch):
    module = installer_module()
    executable = module.environment_python(tmp_path)
    executable.parent.mkdir()
    executable.write_text("existing interpreter")
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    wheel = tmp_path / "verified.whl"
    monkeypatch.setattr(module, "download_verified_pip", lambda path: wheel)
    calls = []
    monkeypatch.setattr(module, "run", lambda command: calls.append(command))
    assert module.ensure_environment(tmp_path) == executable
    assert calls[0][0] == executable and calls[0][-1] == wheel
    assert "runpy" in calls[0][2]
    assert sentinel.read_text() == "keep"


def test_installer_logs_original_command_failure(tmp_path, monkeypatch, capsys):
    module = installer_module()
    module.LOG_FILE = tmp_path / "setup.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        module.run(
            [
                sys.executable,
                "-c",
                "import sys; print('original failure detail'); sys.exit(7)",
            ]
        )
    assert error.value.returncode == 7
    assert "original failure detail" in module.LOG_FILE.read_text()
    assert "original failure detail" in capsys.readouterr().out


def test_notebook_python_cells_compile_and_use_subprocess():
    notebook = json.loads((ROOT / "notebooks/Task1_Colab.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook_cell", "exec")
    assert any(".venv-colab/bin/python" in "".join(c["source"]) for c in notebook["cells"])
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_task1.sh")], check=True)

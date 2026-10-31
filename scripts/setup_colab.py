"""Install the assignment in an isolated environment, then verify it.

Default: CUDA 12.8 PyTorch/torchvision pair for Colab. --cpu is solely for
local implementation tests; it does not validate GPU training or memory use.
The notebook kernel and Colab's preinstalled packages are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
TORCH = "2.9.1"
TORCHVISION = "0.24.1"


def run(command, *, cwd=ROOT):
    print("+ " + shlex.join([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


def environment_python(directory):
    return directory / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )


def validate_existing(directory, cpu):
    """Refuse to reuse a shared environment or silently switch its interpreter."""
    if not directory.exists():
        return
    cfg = directory / "pyvenv.cfg"
    if not cfg.exists():
        raise ValueError(
            f"{directory} exists and is not a virtual environment; choose --venv PATH"
        )
    settings = dict(
        line.split(" = ", 1) for line in cfg.read_text().splitlines() if " = " in line
    )
    if settings.get("include-system-site-packages", "").lower() != "false":
        raise ValueError("Choose an isolated venv without Colab's shared site-packages")
    version = subprocess.check_output(
        [
            str(environment_python(directory)),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError(
            f"Existing venv uses Python {version}; choose a new --venv PATH"
        )
    manifest = directory / "pa3_setup.json"
    if manifest.exists() and json.loads(manifest.read_text())["cpu"] != cpu:
        raise ValueError(
            "CPU/CUDA profile differs from this venv; choose another --venv PATH"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU validation only; never use for A100 training",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest, retaining import/operator/CUDA checks",
    )
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("The pinned Colab stack supports Python 3.11–3.13")
    directory = (
        args.venv or ROOT / (".venv-cpu" if args.cpu else ".venv-colab")
    ).resolve()
    validate_existing(directory, args.cpu)
    if not args.cpu:
        run(["nvidia-smi"])
    if not directory.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(directory)
    python = environment_python(directory)
    build = "cpu" if args.cpu else "cu128"
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--index-url",
            f"https://download.pytorch.org/whl/{build}",
            f"torch=={TORCH}+{build}",
            f"torchvision=={TORCHVISION}+{build}",
        ]
    )
    run([python, "-m", "pip", "install", "-r", ROOT / "requirements-colab.txt"])
    run([python, "-m", "pip", "install", "-e", f"{ROOT}[test]"])
    run([python, "-m", "pip", "check"])
    doctor = [python, "-m", "atml_pa3.cli", "doctor"]
    if not args.cpu:
        doctor.append("--require-cuda")
    run(doctor)
    if not args.skip_tests:
        run([python, "-m", "pytest", "-q"])
    (directory / "pa3_setup.json").write_text(
        json.dumps(
            dict(
                python=sys.version.split()[0],
                cpu=args.cpu,
                torch=TORCH,
                torchvision=TORCHVISION,
                index=f"https://download.pytorch.org/whl/{build}",
                tests_run=not args.skip_tests,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"\nSetup passed. In EVERY later %%bash cell, activate:\nsource {directory}/bin/activate",
        flush=True,
    )
    if args.cpu:
        print("CPU checks passed; CUDA/A100 execution remains unverified.")


if __name__ == "__main__":
    main()

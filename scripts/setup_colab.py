"""Install the assignment in an isolated environment, then verify it.

Default: CUDA 12.8 PyTorch/torchvision pair for Colab. --cpu is solely for
local implementation tests; it does not validate GPU training or memory use.
The notebook kernel and Colab's preinstalled packages are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]
TORCH = "2.9.1"
TORCHVISION = "0.24.1"
PIP_VERSION = "26.2.1"
LOG_FILE = None


def log(message):
    print(message, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as output:
            output.write(message + "\n")


def run(command, *, cwd=ROOT):
    log("+ " + shlex.join([str(value) for value in command]))
    with subprocess.Popen(
        [str(value) for value in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        for line in process.stdout:
            log(line.rstrip("\n"))
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def download_verified_pip(directory):
    """Obtain a universal wheel using TLS and its SHA-256 from PyPI metadata."""
    metadata_url = f"https://pypi.org/pypi/pip/{PIP_VERSION}/json"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    filename = f"pip-{PIP_VERSION}-py3-none-any.whl"
    entry = next(
        (item for item in metadata["urls"] if item["filename"] == filename), None
    )
    if entry is None:
        raise RuntimeError(f"PyPI did not publish the expected wheel: {filename}")
    target = directory / ".bootstrap" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        data = target.read_bytes()
    else:
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != entry["digests"]["sha256"]:
        raise RuntimeError(
            "pip wheel SHA-256 verification failed; no bootstrap code was executed"
        )
    target.write_bytes(data)
    log(f"Verified {filename} against its PyPI SHA-256")
    return target


def ensure_environment(directory):
    # Debian/Colab system interpreters may contain neither ensurepip nor pip.
    # Create the isolated interpreter without invoking ensurepip at all.
    python = environment_python(directory)
    if not python.exists():
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    probe = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        wheel = download_verified_pip(directory)
        # Import only the checksum-verified universal wheel, and install pip into
        # the venv itself. No system package, apt package, or trust setting changes.
        bootstrap = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('pip', run_name='__main__')"
        run([python, "-c", bootstrap, wheel, "install", wheel])
    return python


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
    global LOG_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--log", type=Path, default=ROOT / "setup_colab.log")
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
    LOG_FILE = args.log.resolve()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"Starting setup with {sys.executable} (Python {sys.version.split()[0]})")
    log(f"Full setup log: {LOG_FILE}")
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("The pinned Colab stack supports Python 3.11–3.13")
    directory = (
        args.venv or ROOT / (".venv-cpu" if args.cpu else ".venv-colab")
    ).resolve()
    validate_existing(directory, args.cpu)
    if not args.cpu:
        run(["nvidia-smi"])
    python = ensure_environment(directory)
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
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        log(f"SETUP FAILED. Read the first error above and the full log: {LOG_FILE}")
        raise SystemExit(1)

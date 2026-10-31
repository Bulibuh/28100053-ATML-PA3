"""Notebook-safe subprocess output streaming; import with runpy, no kernel installs."""
from pathlib import Path
import os
import signal
import subprocess
import time


def run_logged(command, cwd, log_path=None):
    cwd = Path(cwd)
    log_path = Path(log_path) if log_path else cwd / "logs" / f"run-{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Log: {log_path}", flush=True)
    tail = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(arg) for arg in command], cwd=cwd,
            env=dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
                tail = tail[-40:]
            status = process.wait()
        except BaseException:
            # Interrupt only the process group created for this invocation.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command, output="".join(tail))
    return log_path


def make_runner(project, python):
    def pa3(*args):
        return run_logged([str(python), "-u", "-m", "atml_pa3.cli", *args], project)
    return pa3

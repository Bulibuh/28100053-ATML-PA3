"""Visible CLI stages and liveness reports; elapsed time is not completion progress."""
from contextlib import contextmanager
import sys
import threading
import time
from tqdm.auto import tqdm


def message(text):
    tqdm.write(str(text), file=sys.stderr)
    sys.stderr.flush()


@contextmanager
def stage(label, interval=15):
    started = time.monotonic()
    done = threading.Event()
    message(f"START: {label}")
    def heartbeat():
        while not done.wait(interval):
            message(f"WORKING: {label} ({time.monotonic() - started:.0f}s elapsed)")
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        message(f"FAILED: {label} after {time.monotonic() - started:.1f}s; see traceback below")
        raise
    else:
        message(f"DONE: {label} ({time.monotonic() - started:.1f}s)")
    finally:
        done.set()
        worker.join()

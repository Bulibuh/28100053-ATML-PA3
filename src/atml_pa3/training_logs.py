"""Local per-update journals; publish standalone logs only after successful training."""
import json
import tempfile
import shutil
import time
from pathlib import Path
from .common import ROOT, append_json, json_default


class TrainingLogs:
    NAMES = ('train', 'overflow', 'checkpoints')

    def __init__(self, method, initial=None, local_root=None):
        local_root = Path(local_root or ROOT / 'logs/training')
        local_root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=f'{method}-', dir=local_root))
        self.rows = {name: list((initial or {}).get(name, [])) for name in self.NAMES}
        for name in self.NAMES:
            self._write(self.directory / f'{name}.jsonl', self.rows[name])

    @staticmethod
    def _write(path, rows):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w') as f:
            for row in rows:
                f.write(json.dumps(row, default=json_default) + '\n')
        temporary.replace(path)

    def append(self, name, row):
        append_json(self.directory / f'{name}.jsonl', row)
        self.rows[name].append(row)

    def snapshot(self):
        # Copy lists so subsequent appends cannot change an existing snapshot.
        return {name: list(rows) for name, rows in self.rows.items()}

    def publish(self, destination):
        """One bulk replacement per log, called only at successful completion."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in self.rows.items():
            if rows or name == 'train':
                path = destination / f'{name}.jsonl'
                if path.exists():
                    shutil.copy2(path, destination / f'{name}_before_sync_{time.time_ns()}.jsonl')
                self._write(path, rows)

"""Fetch the small public inputs actually linked in the assignment PDF.
Content hashes pin the inspected versions. A changed upstream file fails closed;
review the new version before deliberately updating its recorded checksum.
"""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    for item in json.loads((ROOT / "configs/public_inputs.json").read_text()):
        path = ROOT / "data/external" / item["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(
                f'Checksum mismatch for {item["filename"]}; inspect the source version, do not bypass verification'
            )
        if not path.exists():
            path.write_bytes(data)
        print("Verified", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

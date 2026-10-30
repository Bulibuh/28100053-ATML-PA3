"""Download official COCO/VQA annotations with HTTPS verification and safe extraction.
URLs are recorded explicitly; individual servers can change their published paths.
If a URL fails, use the official download page and put the JSONs in --output.
"""

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

URLS = ["https://images.cocodataset.org/annotations/annotations_trainval2014.zip"] + [
    f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_{kind}_{split}_mscoco.zip"
    for kind in ["Questions", "Annotations", "Complementary_Pairs"]
    for split in ["Train", "Val"]
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/external/coco/annotations")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        archive = out / url.rsplit("/", 1)[1]
        if not archive.exists():
            urllib.request.urlretrieve(url, str(archive) + ".tmp")
            Path(str(archive) + ".tmp").replace(archive)
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {archive}, {bad}")
            for entry in z.infolist():
                if not entry.filename.endswith(".json"):
                    continue
                dest = out / Path(entry.filename).name
                if not dest.exists():
                    with z.open(entry) as source, dest.open("wb") as target:
                        shutil.copyfileobj(source, target)
        print("Verified ZIP integrity and extracted JSON:", archive.name)


if __name__ == "__main__":
    main()

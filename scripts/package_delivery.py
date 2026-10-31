"""Create a source/guide/evidence bundle without Git metadata, datasets, or weights."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "atml-pa3-deliverable.zip"
ALLOWED = {"src", "configs", "scripts", "docs", "notebooks", "tests", "evidence"}
TOP_FILES = {
    ".gitignore",
    "README.md",
    "ATTRIBUTIONS.md",
    "pyproject.toml",
    "requirements-colab.txt",
}


def main():
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        candidates = [ROOT / name for name in TOP_FILES]
        candidates.append(ROOT / "data/external/README.md")
        for name in sorted(ALLOWED):
            candidates.extend((ROOT / name).rglob("*"))
        for path in sorted(candidates):
            relative = path.relative_to(ROOT)
            if (
                not path.is_file()
                or "__pycache__" in relative.parts
                or any(part.endswith(".egg-info") for part in relative.parts)
                or path.suffix in {".pyc", ".pt", ".safetensors", ".parquet"}
            ):
                continue
            if (
                relative.parts[0] in ALLOWED
                or str(relative) in TOP_FILES
                or relative == Path("data/external/README.md")
            ):
                archive.write(path, Path("atml-pa3") / relative)
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive integrity failure: {bad}")
        print(
            f"{OUTPUT}: {len(archive.namelist())} files, {OUTPUT.stat().st_size} bytes"
        )


if __name__ == "__main__":
    main()

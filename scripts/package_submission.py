"""Create a portable archive from the staged files in the local Git repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git", default=shutil.which("git") or str(ROOT / ".tools/git/cmd/git.exe"))
    parser.add_argument("--output", type=Path, default=ROOT / "submission/CMPSC448_Midterm1_Submission.zip")
    args = parser.parse_args()
    subprocess.run([args.git, "add", "."], cwd=ROOT, check=True)
    paths = subprocess.check_output([args.git, "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    paths = sorted(path for path in paths if path)
    if "report/project_report.pdf" not in paths:
        raise ValueError("The staged submission has no final PDF")
    manifest = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in paths:
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError(f"Invalid submission path: {relative}")
            if any(part in (".venv", ".git", ".tools", "__pycache__") for part in Path(relative).parts):
                raise ValueError(f"Environment file included in archive: {relative}")
            if relative.startswith("data/raw/"):
                raise ValueError(f"Raw cache included in archive: {relative}")
            content = path.read_bytes()
            manifest[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)
        archive.writestr("SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.output.name}\n", encoding="utf-8")
    print(f"Packaged {len(paths)} staged files: {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()

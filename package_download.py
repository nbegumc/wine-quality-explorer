"""Package reproducible source and results; exclude runtime and hosting metadata."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {".python-deps", ".git", ".venv", "__pycache__", ".openai", ".sites-runtime", ".pytest_cache"}

if __name__ == "__main__":
    target = ROOT / "dist/wine-quality-python.zip"
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        entries = ["README.md", "pyproject.toml", "uv.lock", ".python-version", "requirements.txt",
                   "serve.py", "train.py", "package_download.py",
                   ".gitignore", "src", "tests", "docs", "data", "original", "dist"]
        paths = []
        for entry in entries:
            base = ROOT / entry
            paths.extend(base.rglob("*") if base.is_dir() else [base])
        for path in sorted(paths):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
                continue
            if path == target or path.suffix in {".zip", ".pyc"}:
                continue
            archive.write(path, Path("wine-quality-explorer") / relative)
    print(f"Created {target.name} ({target.stat().st_size:,} bytes)")

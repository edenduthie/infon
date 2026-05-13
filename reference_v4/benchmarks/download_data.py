"""Download FEVER and HoVer benchmark datasets.

FEVER: fact verification with gold sentence-level evidence.
HoVer: multi-hop verification requiring linked document traversal.

Both datasets use Wikipedia as the knowledge source.
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_fever():
    """Download FEVER shared task data (paper split)."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fever_dir = DATA_DIR / "fever"
    fever_dir.mkdir(exist_ok=True)

    base = "https://fever.ai/download/fever"
    files = {
        "train.jsonl": f"{base}/train.jsonl",
        "paper_dev.jsonl": f"{base}/paper_dev.jsonl",
        "paper_test.jsonl": f"{base}/paper_test.jsonl",
    }

    # Also need the Wikipedia dump for evidence pages
    wiki_url = "https://fever.ai/download/fever/wiki-pages.zip"

    for name, url in files.items():
        dest = fever_dir / name
        if dest.exists():
            print(f"  [skip] {dest} already exists")
            continue
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")

    # Wikipedia pages (large — ~1.5GB compressed)
    wiki_zip = fever_dir / "wiki-pages.zip"
    if not wiki_zip.exists() and not (fever_dir / "wiki-pages").exists():
        print(f"  Downloading wiki-pages.zip (~1.5GB)...")
        print(f"  URL: {wiki_url}")
        print(f"  [manual] Download manually if this takes too long:")
        print(f"    curl -L '{wiki_url}' -o '{wiki_zip}'")
        try:
            urllib.request.urlretrieve(wiki_url, wiki_zip)
        except Exception as e:
            print(f"  [warn] Download failed: {e}")
            print(f"  Download manually from {wiki_url}")

    if wiki_zip.exists() and not (fever_dir / "wiki-pages").exists():
        import zipfile
        print("  Extracting wiki-pages.zip...")
        with zipfile.ZipFile(wiki_zip) as zf:
            zf.extractall(fever_dir)


def download_hover():
    """Download HoVer dataset from GitHub."""
    import urllib.request

    hover_dir = DATA_DIR / "hover"
    hover_dir.mkdir(parents=True, exist_ok=True)

    base = "https://raw.githubusercontent.com/hover-nlp/hover/main/data"
    files = {
        "hover_train.json": f"{base}/hover_train.json",
        "hover_dev.json": f"{base}/hover_dev.json",
        # HoVer uses the same Wikipedia dump as FEVER
    }

    for name, url in files.items():
        dest = hover_dir / name
        if dest.exists():
            print(f"  [skip] {dest} already exists")
            continue
        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest}")
        except Exception as e:
            print(f"  [warn] {e}")
            print(f"  Try: curl -L '{url}' -o '{dest}'")


def verify_data():
    """Check what we have."""
    print("\n=== Data inventory ===")
    fever_dir = DATA_DIR / "fever"
    hover_dir = DATA_DIR / "hover"

    for f in sorted(fever_dir.glob("*")) if fever_dir.exists() else []:
        sz = f.stat().st_size / (1024 * 1024)
        print(f"  FEVER: {f.name} ({sz:.1f} MB)")

    for f in sorted(hover_dir.glob("*")) if hover_dir.exists() else []:
        sz = f.stat().st_size / (1024 * 1024)
        print(f"  HoVer: {f.name} ({sz:.1f} MB)")


if __name__ == "__main__":
    print("=== Downloading FEVER ===")
    download_fever()
    print("\n=== Downloading HoVer ===")
    download_hover()
    verify_data()

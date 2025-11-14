"""
Web Image Downloader for Dataset Expansion

This utility searches the web for images per label and downloads them
into your existing dataset structure: ../dataset/<LABEL>.

It supports two backends:
- simple_image_download (preferred if installed)
- duckduckgo_search + requests (fallback)

Usage examples:
  python download_images.py --labels A B C --per_label 150
  python download_images.py --labels A --query_map A:"ASL letter A hand sign" --per_label 200

Notes:
- Results from the public web are noisy. After download, quickly skim and
  delete obvious wrong images to keep the dataset clean.
- You can safely rerun; it skips files that already exist.
"""

import argparse
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List

import threading
from queue import Queue

try:
    # Preferred scraper (no API key needed)
    from simple_image_download import simple_image_download as sid  # type: ignore
    HAS_SIMPLE = True
except Exception:
    HAS_SIMPLE = False

HAS_DDG = False
try:
    # Preferred new package name
    from ddgs import DDGS  # type: ignore
    import requests
    HAS_DDG = True
except Exception:
    try:
        # Older package name
        from duckduckgo_search import DDGS  # type: ignore
        import requests  # type: ignore
        HAS_DDG = True
    except Exception:
        HAS_DDG = False


DEFAULT_DATASET_DIR = "../dataset"
DEFAULT_PER_LABEL = 120


def ensure_label_dir(base_dir: Path, label: str) -> Path:
    out = base_dir / label
    out.mkdir(parents=True, exist_ok=True)
    return out


def sanitize_filename(url: str) -> str:
    # Stable short name based on URL to avoid duplicates
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"img_{h}.jpg"


def download_via_simple(label: str, query: str, per_label: int, out_dir: Path) -> int:
    if not HAS_SIMPLE:
        return 0
    downloader_factory = None
    if hasattr(sid, "simple_image_download"):
        downloader_factory = getattr(sid, "simple_image_download")
    elif hasattr(sid, "SimpleImageDownload"):
        downloader_factory = getattr(sid, "SimpleImageDownload")
    elif callable(getattr(sid, "SimpleImageDownload", None)):
        downloader_factory = getattr(sid, "SimpleImageDownload")

    if downloader_factory is None:
        print("simple_image_download module is installed but no downloader class was found.")
        return 0

    try:
        response = downloader_factory()
    except Exception:
        print("Failed to instantiate simple_image_download downloader. Skipping this backend.")
        return 0

    # simple_image_download writes to its own folder; we intercept by moving/renaming
    tmp_root = out_dir.parent / f"__tmp_simple_{label}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        response.download(keywords=query, limit=per_label, extensions={".jpg", ".jpeg", ".png"}, print_urls=False, output_directory=str(tmp_root))
    except Exception:
        # Library can be noisy; continue gracefully
        pass

    # Move results from tmp to out_dir, rename to stable names
    moved = 0
    for p in tmp_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                # Build a deterministic name
                new_name = f"simple_{label}_{int(time.time()*1000)}_{moved:04d}{p.suffix.lower()}"
                dest = out_dir / new_name
                p.replace(dest)
                moved += 1
            except Exception:
                continue

    # Cleanup tmp folder
    try:
        for p in sorted(tmp_root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                p.rmdir()
        tmp_root.rmdir()
    except Exception:
        pass

    return moved


def fetch_ddg_urls(query: str, max_results: int) -> List[str]:
    urls: List[str] = []
    if not HAS_DDG:
        return urls
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(keywords=query, max_results=max_results, safesearch="moderate"):
                url = r.get("image")
                if url:
                    urls.append(url)
    except Exception:
        pass
    return urls


def threaded_download(urls: List[str], out_dir: Path, max_threads: int = 16, timeout: int = 10) -> int:
    if not HAS_DDG:
        return 0
    q: "Queue[str]" = Queue()
    for u in urls:
        q.put(u)

    downloaded = 0
    downloaded_lock = threading.Lock()

    def worker():
        nonlocal downloaded
        session = requests.Session()
        while True:
            try:
                url = q.get_nowait()
            except Exception:
                break
            try:
                resp = session.get(url, timeout=timeout, stream=True)
                if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                    name = sanitize_filename(url)
                    dest = out_dir / name
                    if dest.exists():
                        q.task_done()
                        continue
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    with downloaded_lock:
                        downloaded += 1
            except Exception:
                pass
            finally:
                q.task_done()

    threads = []
    for _ in range(min(max_threads, len(urls))):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    q.join()
    for t in threads:
        t.join(timeout=1)

    return downloaded


def download_for_label(base_dir: Path, label: str, query: str, per_label: int) -> int:
    out_dir = ensure_label_dir(base_dir, label)
    print(f"\nLabel '{label}': searching for ~{per_label} images with query: {query}")

    total = 0
    # Try preferred backend first
    if HAS_SIMPLE:
        got = download_via_simple(label, query, per_label, out_dir)
        total += got
        remaining = max(0, per_label - got)
    else:
        remaining = per_label

    # Fallback: DDG URL fetch + requests download
    if remaining > 0 and HAS_DDG:
        urls = fetch_ddg_urls(query, remaining * 3)  # Oversample; many links may fail
        got = threaded_download(urls, out_dir)
        total += got

    print(f"Saved {total} images to {out_dir}")
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Download web images into dataset/<LABEL> to expand training data.")
    parser.add_argument("--labels", nargs="+", help="Labels to download (e.g., A B C)")
    parser.add_argument("--per_label", type=int, default=DEFAULT_PER_LABEL, help="Target images per label")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR, help="Base dataset directory (default: ../dataset)")
    parser.add_argument("--query_map", nargs="*", help='Optional custom queries per label, format: LABEL:"custom search query"')
    return parser.parse_args()


def build_query_map(labels: List[str], raw_map: List[str] | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    provided: Dict[str, str] = {}
    if raw_map:
        for item in raw_map:
            # Expect LABEL:"query with spaces"
            try:
                key, val = item.split(":", 1)
                provided[key.strip()] = val.strip().strip('"').strip("'")
            except ValueError:
                continue
    for lbl in labels:
        if lbl in provided:
            mapping[lbl] = provided[lbl]
        else:
            # Reasonable default query for ASL alphabet letters
            mapping[lbl] = f"ASL letter {lbl} hand sign"
    return mapping


def main():
    args = parse_args()
    if not args.labels:
        print("Error: --labels is required (e.g., --labels A B C)")
        sys.exit(1)

    labels = [s.strip() for s in args.labels if s.strip()]
    base_dir = Path(args.dataset_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    query_map = build_query_map(labels, args.query_map)

    print("Downloader backends:")
    print(f" - simple_image_download: {'available' if HAS_SIMPLE else 'missing'}")
    print(f" - duckduckgo_search: {'available' if HAS_DDG else 'missing'}")
    if not HAS_SIMPLE and not HAS_DDG:
        print("\nNo downloader backend available.")
        print("Install one of the following and re-run:")
        print("  pip install simple_image_download")
        print("  pip install duckduckgo-search requests")
        sys.exit(1)

    total_all = 0
    for lbl in labels:
        total_all += download_for_label(base_dir, lbl, query_map[lbl], args.per_label)

    print(f"\nDone. Downloaded {total_all} images across {len(labels)} labels.")


if __name__ == "__main__":
    main()


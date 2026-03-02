"""
sync_incremental.py — Safe Incremental Data Sync Script
========================================================
ดึงข้อมูลเฉพาะหน้าที่เปลี่ยนแปลงแล้ว upsert เข้า ChromaDB
ระบบหลักที่รันอยู่ "ไม่กระทบ" ระหว่างรัน script นี้

Usage:
    python scripts/sync_incremental.py
    python scripts/sync_incremental.py --dry-run        # แค่ดูว่าจะอัปเดตหน้าไหน (ไม่แตะ DB)
    python scripts/sync_incremental.py --target-url "http://10.192.133.33/smc/index.php?..."  # อัปเดตหน้าเดียว
"""

from __future__ import annotations

import sys
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("IncrementalSync")

# ── Constants ─────────────────────────────────────────────────────────────────
STATE_PATH   = ROOT / "data" / "state.json"
CONFIG_PATH  = ROOT / "configs" / "config.yaml"
HEADERS = {
    "User-Agent": "rag-incremental-sync/1.0 (+internal)",
    "Accept": "text/html,application/xhtml+xml",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"pages": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def fetch_page(url: str, timeout: int = 15) -> Optional[str]:
    """ดึง HTML จาก URL ถ้าล้มเหลวคืน None"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.encoding = r.apparent_encoding or "utf-8"
        if r.ok:
            return r.text
        log.warning(f"HTTP {r.status_code} for {url}")
    except Exception as e:
        log.error(f"Fetch failed for {url}: {e}")
    return None


def extract_text(html: str, base_url: str = "") -> str:
    """แปลง HTML → plain text (เน้น article body)"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer"]):
        tag.decompose()
    # เน้น content area ก่อน
    body = (
        soup.select_one("div.item-page")
        or soup.select_one("div#content")
        or soup.select_one("article")
        or soup.body
    )
    return (body or soup).get_text("\n", strip=True)


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """แบ่ง text เป็น chunks ขนาดเท่ากัน"""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def discover_urls(cfg: dict) -> list[str]:
    """ค้น URLs บน Sitemap / หน้าหลัก (ใช้แค่ 1 level crawl)"""
    domain = cfg["web"]["domain"]
    start_urls: list[str] = cfg["web"]["start_urls"]
    allowed_paths: list[str] = cfg["web"].get("allowed_paths", ["/"])
    deny_ext: list[str] = cfg["web"].get("deny_extensions", [])

    collected = set()
    for start in start_urls:
        html = fetch_page(start)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # สร้าง absolute URL
            if href.startswith("http"):
                full = href
            elif href.startswith("/"):
                full = f"http://{domain}{href}"
            else:
                continue
            # กรอง
            if domain not in full:
                continue
            if any(full.endswith(ext) for ext in deny_ext):
                continue
            if any(p in full for p in allowed_paths):
                collected.add(full)

    log.info(f"Discovered {len(collected)} URLs")
    return sorted(collected)


def upsert_to_chroma(cfg: dict, url: str, chunks: list[str]) -> int:
    """Upsert chunks เข้า ChromaDB — คืนจำนวน chunks ที่อัปเดต"""
    from src.vectorstore.chroma import ChromaVectorStore

    vs_cfg = cfg["vectorstore"]
    vs = ChromaVectorStore(
        persist_dir=str(ROOT / vs_cfg["persist_dir"]),
        collection_name=vs_cfg["collection_name"],
        embedding_model=vs_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        bm25_path=str(ROOT / "data" / "bm25_index.json"),
    )

    # ลบ chunks เก่าของ URL นี้ก่อน (clean slate per-URL)
    try:
        vs.delete_by_url(url)
        log.debug(f"Deleted old chunks for {url}")
    except Exception as e:
        log.warning(f"Could not delete old chunks for {url}: {e}")

    # Upsert chunks ใหม่
    ids = [f"{compute_hash(url)}_{i}" for i in range(len(chunks))]
    metadatas = [{"url": url, "chunk_index": i, "updated_at": time.time()} for i in range(len(chunks))]
    vs.upsert(ids=ids, texts=chunks, metadatas=metadatas)
    return len(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

def sync_url(url: str, state: dict, cfg: dict, dry_run: bool = False) -> str:
    """
    Sync หน้าเดียว คืน: 'updated' | 'skipped' | 'error'
    """
    html = fetch_page(url)
    if html is None:
        return "error"

    text = extract_text(html, base_url=url)
    new_hash = compute_hash(text)
    old_hash = state.get("pages", {}).get(url, {}).get("content_hash")

    if old_hash == new_hash:
        log.info(f"[SKIP]    {url}  (no change)")
        return "skipped"

    log.info(f"[CHANGED] {url}  hash {old_hash[:8] if old_hash else 'NEW'}→{new_hash[:8]}")

    if not dry_run:
        chunks = chunk_text(text, cfg["chunk"]["chunk_size"], cfg["chunk"]["overlap"])
        n = upsert_to_chroma(cfg, url, chunks)
        # อัปเดต state
        state.setdefault("pages", {})[url] = {
            "content_hash": new_hash,
            "indexed_hash": new_hash,
            "chunk_count": n,
            "last_synced": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        log.info(f"[UPDATED] {url}  upserted {n} chunks")
    else:
        log.info(f"[DRY-RUN] Would upsert {url}")

    return "updated"


def main():
    parser = argparse.ArgumentParser(description="Incremental RAG sync")
    parser.add_argument("--dry-run", action="store_true", help="แค่ดูว่าจะอัปเดตอะไรบ้าง ไม่แตะ DB")
    parser.add_argument("--target-url", type=str, default=None, help="อัปเดตเฉพาะ URL นี้")
    args = parser.parse_args()

    cfg   = load_config()
    state = load_state()

    if args.target_url:
        urls = [args.target_url]
    else:
        log.info("Discovering URLs...")
        urls = discover_urls(cfg)

    stats = {"updated": 0, "skipped": 0, "error": 0}

    for i, url in enumerate(urls, 1):
        log.info(f"[{i}/{len(urls)}] Processing: {url}")
        result = sync_url(url, state, cfg, dry_run=args.dry_run)
        stats[result] += 1
        time.sleep(cfg["web"].get("rate_limit_sec", 0.5))

    if not args.dry_run:
        save_state(state)

    log.info("=" * 60)
    log.info(f"Sync Complete — Updated: {stats['updated']} | Skipped: {stats['skipped']} | Error: {stats['error']}")


if __name__ == "__main__":
    main()

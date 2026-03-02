"""
swap_collection.py — Blue-Green Collection Swap Script
=======================================================
สร้าง Collection ใหม่ (Shadow) → ทดสอบ → สลับ → ระบบหลักไม่ดาวน์เลย

Flow:
    1. สร้าง shadow collection ชื่อ: smc_web_shadow
    2. Crawl + Ingest ข้อมูลใหม่ทั้งหมดเข้า shadow
    3. รัน Smoke Test (คำถาม baseline 5 ข้อ) ทดสอบถังใหม่
    4. ถ้าผ่าน → เขียน config.yaml ให้ชี้ไปถังใหม่ (backup เก่าไว้)
    5. ถ้าไม่ผ่าน → ออก EXIT_CODE 1 (ไม่แตะระบบหลัก)

Usage:
    python scripts/swap_collection.py --help
    python scripts/swap_collection.py --build-only     # ingest เข้า shadow แต่ยังไม่สลับ
    python scripts/swap_collection.py --test-only      # แค่ทดสอบ shadow ที่มีอยู่แล้ว
    python scripts/swap_collection.py --swap-only      # สลับอย่างเดียว (ถ้า shadow พร้อมแล้ว)
    python scripts/swap_collection.py                  # ทำครบ 3 ขั้นตอน
"""

from __future__ import annotations

import sys
import json
import time
import hashlib
import argparse
import shutil
import logging
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("BlueGreenSwap")

CONFIG_PATH = ROOT / "configs" / "config.yaml"

# ── Baseline Smoke Tests ───────────────────────────────────────────────────────
# คำถามที่ระบบ "ควรค้นเจอ" ใน Collection ที่ดี
# ปรับได้ตามข้อมูลจริงของพี่
SMOKE_TESTS = [
    "5ส",
    "contact",
    "SMC",
    "วิธีใช้",
    "manual",
]
MIN_RESULTS_REQUIRED = 1   # แต่ละคำถามต้องได้ผลลัพธ์อย่างน้อย N ข้อ
MIN_SCORE_REQUIRED   = 0.1 # score ขั้นต่ำ


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config_with_collection(collection_name: str) -> None:
    """อัปเดต collection_name ใน config.yaml (backup ก่อน)"""
    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, backup)
    log.info(f"Config backed up to {backup}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = f.read()

    # แทนที่บรรทัด collection_name
    lines = raw.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("collection_name:"):
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f"collection_name: {collection_name}")
        else:
            new_lines.append(line)

    CONFIG_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    log.info(f"config.yaml → collection_name: {collection_name}")


def get_vectorstore(cfg: dict, collection_name: Optional[str] = None):
    """สร้าง ChromaVectorStore instance"""
    from src.vectorstore.chroma import ChromaVectorStore
    vs_cfg = cfg["vectorstore"]
    name = collection_name or vs_cfg["collection_name"]
    return ChromaVectorStore(
        persist_dir=str(ROOT / vs_cfg["persist_dir"]),
        collection_name=name,
        embedding_model=vs_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        bm25_path=str(ROOT / "data" / f"bm25_{name}.json"),
    )


# ── Phase 1: Build Shadow ─────────────────────────────────────────────────────

def build_shadow(cfg: dict) -> int:
    """
    Crawl & Ingest ข้อมูลทั้งหมดเข้า shadow collection
    คืนจำนวน URLs ที่ประมวลผลสำเร็จ
    """
    import requests
    from bs4 import BeautifulSoup

    shadow_name = cfg["vectorstore"]["collection_name"] + "_shadow"
    log.info(f"Building shadow collection: {shadow_name}")

    # สร้าง VectorStore เปล่าสำหรับ shadow
    shadow_vs = get_vectorstore(cfg, shadow_name)

    domain = cfg["web"]["domain"]
    start_urls: list[str] = cfg["web"]["start_urls"]
    deny_ext: list[str] = cfg["web"].get("deny_extensions", [])
    rate  = cfg["web"].get("rate_limit_sec", 0.5)
    chunk_size = cfg["chunk"]["chunk_size"]
    overlap    = cfg["chunk"]["overlap"]

    headers = {"User-Agent": "rag-bg-sync/1.0", "Accept": "text/html"}

    def fetch(url: str) -> Optional[str]:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text if r.ok else None
        except Exception as e:
            log.warning(f"Fetch error {url}: {e}")
            return None

    def extract(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        body = (soup.select_one("div.item-page") or
                soup.select_one("div#content") or
                soup.select_one("article") or soup.body)
        return (body or soup).get_text("\n", strip=True)

    def chunk(text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            c = " ".join(words[i: i + chunk_size])
            if c.strip():
                chunks.append(c)
            if i + chunk_size >= len(words):
                break
        return chunks

    # Discover URLs
    all_urls: set[str] = set()
    for start in start_urls:
        html = fetch(start)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full = href if href.startswith("http") else (f"http://{domain}{href}" if href.startswith("/") else None)
            if full and domain in full and not any(full.endswith(e) for e in deny_ext):
                all_urls.add(full)

    log.info(f"Found {len(all_urls)} URLs for shadow build")
    ok_count = 0

    for i, url in enumerate(sorted(all_urls), 1):
        log.info(f"[{i}/{len(all_urls)}] {url}")
        html = fetch(url)
        if not html:
            continue
        text = extract(html)
        chunks = chunk(text)
        if not chunks:
            continue
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        ids = [f"{url_hash}_{j}" for j in range(len(chunks))]
        metas = [{"url": url, "chunk_index": j, "built_at": time.time()} for j in range(len(chunks))]
        try:
            shadow_vs.upsert(ids=ids, texts=chunks, metadatas=metas)
            ok_count += 1
        except Exception as e:
            log.error(f"Upsert failed: {e}")
        time.sleep(rate)

    log.info(f"Shadow build complete — {ok_count}/{len(all_urls)} pages indexed")
    return ok_count


# ── Phase 2: Smoke Test ───────────────────────────────────────────────────────

def run_smoke_tests(cfg: dict, collection_name: Optional[str] = None) -> bool:
    """
    รัน Baseline Smoke Test ตรวจสอบว่า shadow ตอบได้
    คืน True = ผ่าน, False = ไม่ผ่าน
    """
    name = collection_name or (cfg["vectorstore"]["collection_name"] + "_shadow")
    log.info(f"Running smoke tests on: {name}")
    vs = get_vectorstore(cfg, name)

    passed = 0
    for q in SMOKE_TESTS:
        results = vs.query(q, top_k=MIN_RESULTS_REQUIRED)
        top_score = results[0].score if results else 0.0
        status = "✅ PASS" if (results and top_score >= MIN_SCORE_REQUIRED) else "❌ FAIL"
        log.info(f"  {status}  '{q}' → top_score={top_score:.3f}, results={len(results)}")
        if results and top_score >= MIN_SCORE_REQUIRED:
            passed += 1

    pass_rate = passed / len(SMOKE_TESTS)
    log.info(f"Smoke Test: {passed}/{len(SMOKE_TESTS)} passed ({pass_rate*100:.0f}%)")
    return pass_rate >= 0.6   # ผ่านอย่างน้อย 60%


# ── Phase 3: Swap ─────────────────────────────────────────────────────────────

def swap(cfg: dict) -> None:
    """
    สลับ collection_name ใน config.yaml
    shadow → active, active → old
    """
    current  = cfg["vectorstore"]["collection_name"]
    shadow   = current + "_shadow"
    old_name = current + "_old"

    # Rename logic ผ่าน config (ChromaDB ไม่รองรับ rename collection โดยตรง)
    # วิธีที่ปลอดภัยที่สุดคือ: แก้ config.yaml ให้ชี้ไป shadow แทน
    # (ถังเก่ายังอยู่ใน DB แค่ไม่ได้ใช้)
    save_config_with_collection(shadow)

    log.info("=" * 60)
    log.info(f"✅ SWAP COMPLETE")
    log.info(f"   Active collection : {shadow}")
    log.info(f"   Old collection    : {current}  (ยังอยู่ใน DB ถ้าจะ rollback)")
    log.info(f"   Config backup     : configs/config.yaml.bak")
    log.info("=" * 60)
    log.info("⚠️  Restart uvicorn เพื่อให้ API โหลด config ใหม่:")
    log.info("    pkill -f uvicorn && uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload")


def rollback(cfg: dict) -> None:
    """คืนกลับ config.yaml จาก backup"""
    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    if not backup.exists():
        log.error("ไม่พบ config.yaml.bak ไม่สามารถ rollback ได้")
        sys.exit(1)
    shutil.copy2(backup, CONFIG_PATH)
    log.info(f"✅ Rollback สำเร็จ — คืนค่า config.yaml จาก {backup}")
    log.info("⚠️  Restart uvicorn เพื่อโหลด config เดิม")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blue-Green Collection Swap")
    parser.add_argument("--build-only",  action="store_true", help="เฉพาะ ingest เข้า shadow")
    parser.add_argument("--test-only",   action="store_true", help="เฉพาะ run smoke test บน shadow")
    parser.add_argument("--swap-only",   action="store_true", help="เฉพาะ swap config")
    parser.add_argument("--rollback",    action="store_true", help="คืนกลับ config เดิม (จาก .bak)")
    args = parser.parse_args()

    cfg = load_config()

    if args.rollback:
        rollback(cfg)
        return

    if args.swap_only:
        swap(cfg)
        return

    if args.test_only:
        ok = run_smoke_tests(cfg)
        sys.exit(0 if ok else 1)

    if args.build_only:
        build_shadow(cfg)
        return

    # ── Full Pipeline ──────────────────────────────────────────────────────────
    log.info("Starting Full Blue-Green Pipeline (Build → Test → Swap)")

    # Phase 1
    n = build_shadow(cfg)
    if n == 0:
        log.error("Shadow build failed — no pages indexed. Aborting.")
        sys.exit(1)

    # Phase 2
    ok = run_smoke_tests(cfg)
    if not ok:
        log.error("Smoke tests FAILED — ระบบหลักยังใช้ collection เดิม ไม่มีการเปลี่ยนแปลง")
        sys.exit(1)

    # Phase 3
    swap(cfg)


if __name__ == "__main__":
    main()

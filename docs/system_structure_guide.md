# โครงสร้างข้อมูลระบบ RAG — คู่มือสำหรับนักพัฒนา

> อัปเดตล่าสุด: 2026-03-04

เอกสารนี้อธิบาย **ไฟล์ข้อมูลสำคัญ** แต่ละตัว ว่ามีโครงสร้างยังไง และถ้าอยากแก้/เพิ่มข้อมูล ต้องแก้ตรงไหน

---

## ภาพรวม Folder Structure

```
rag_web/
├── configs/
│   └── config.yaml          ← ตั้งค่าหลักของระบบ (แก้ที่นี่ที่เดียว)
├── data/
│   ├── aliases.json          ← แมป ชื่ออุปกรณ์/คำย่อ → ชื่อเต็ม
│   ├── state.json            ← บันทึก hash ของทุก URL ที่ index แล้ว
│   ├── bm25_index.json       ← Keyword search index (auto-generated)
│   ├── metrics.csv           ← Log ทุก request ที่เข้าระบบ
│   ├── records/              ← ฐานข้อมูลผู้ติดต่อ (JSON)
│   ├── sessions/             ← บันทึก session ของผู้ใช้แต่ละคน
│   ├── raw/                  ← HTML ดิบที่ crawl มาจากเว็บ
│   ├── processed/            ← JSON ที่สะอาดแล้ว พร้อม index
│   └── vectorstore/          ← ChromaDB files (Vector Database)
├── src/
│   ├── core/chat_engine.py   ← หัวสมองหลัก (routing, logic)
│   ├── ai/                   ← LLM, normalizer, classifier, router
│   ├── vectorstore/          ← ChromaDB + BM25 wrapper
│   ├── ingest/               ← Crawler, parser, indexer
│   └── api/server.py         ← FastAPI endpoints
└── scripts/
    ├── sync_incremental.py   ← อัปเดตข้อมูลแบบ incremental
    └── swap_collection.py    ← Blue-green rebuild
```

---

## 1. `configs/config.yaml` — ตั้งค่าหลัก

**ถ้าอยากเปลี่ยนอะไร แก้ที่นี่ที่เดียวครับ**

```yaml
web:
  domain: "10.192.133.33"       # ← IP ของเว็บ SMC
  start_urls:
    - "http://10.192.133.33/smc"  # ← จุดเริ่มต้น crawl
  crawl_depth: 3                  # ← ลึกแค่ไหน
  rate_limit_sec: 0.3             # ← หน่วงเวลาระหว่างดึงหน้า

vectorstore:
  collection_name: smc_web        # ← ชื่อ collection ใน ChromaDB
  embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

llm:
  provider: ollama
  model: llama3.2:3b              # ← เปลี่ยน model ที่นี่
  base_url: http://localhost:11434

retrieval:
  top_k: 3                        # ← ดึงผลลัพธ์กี่ชิ้น
  score_threshold: 0.2            # ← ตัดผลที่ score ต่ำกว่านี้ออก
```

**แก้แล้วต้อง Restart:**
```bash
pkill -f uvicorn
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. `data/aliases.json` — ชื่อย่อ / ชื่อเล่นอุปกรณ์

ใช้สำหรับให้ระบบเข้าใจได้ว่า `ne8000` = `huawei ne8000` = `huawei netengine 8000`

```json
{
  "ne8000": ["huawei ne8000", "huawei netengine 8000", "ne 8000"],
  "olt":    ["olt c300", "olt c320", "c6xx", "zte olt"],
  "bras":   ["bras", "broadband remote access server"]
}
```

**ถ้าอยากเพิ่ม alias ใหม่:**
1. เปิดไฟล์ `data/aliases.json`
2. เพิ่มบรรทัดใหม่ตามรูปแบบด้านบน
3. Restart ระบบ (ระบบ reload ไฟล์นี้ตอน startup)

---

## 3. `data/records/` — ฐานข้อมูลผู้ติดต่อ

โครงสร้างไฟล์ JSON แต่ละไฟล์ใน `records/`:

```json
[
  {
    "name": "นายสมชาย ใจดี",
    "team": "NOC",
    "department": "ศูนย์ควบคุมเครือข่าย",
    "phone": "02-123-4567",
    "ext": "101",
    "email": "somchai@nt.com",
    "location": "กรุงเทพ"
  }
]
```

**ถ้าอยากเพิ่ม/แก้ผู้ติดต่อ:**
1. เปิดไฟล์ใน `data/records/` ที่เกี่ยวข้อง
2. แก้ไข JSON ตามรูปแบบด้านบน
3. ไม่ต้อง Restart — ระบบอ่านไฟล์นี้ตอนถามเจอ

---

## 4. `data/state.json` — ตัวติดตาม URL (Hash Registry)

ระบบใช้ไฟล์นี้เช็คว่า "หน้าไหนเปลี่ยนแล้ว" เพื่อไม่ต้อง re-index ทุกหน้าทุกครั้ง

```json
{
  "pages": {
    "http://10.192.133.33/smc/index.php?...": {
      "indexed_hash": "2bbc46fc5f9f...",
      "content_hash": "2bbc46fc5f9f...",
      "chunk_count": 3,
      "last_synced": "2026-03-02T14:39:00"
    }
  }
}
```

**ถ้าอยากบังคับให้ re-index หน้าใดหน้าหนึ่ง:**
```bash
# วิธีง่ายสุด: ลบ hash ของ URL นั้นออกจาก state.json
python3 -c "
import json
state = json.load(open('data/state.json'))
url = 'http://10.192.133.33/smc/index.php?...'
state['pages'].pop(url, None)
json.dump(state, open('data/state.json','w'), ensure_ascii=False, indent=2)
print('Done')
"
# แล้วรัน sync อีกรอบ
python scripts/sync_incremental.py --target-url "http://..."
```

---

## 5. `data/processed/` — บทความที่ประมวลผลแล้ว

ทุกหน้าเว็บที่ crawl มาจะถูกบันทึกเป็น `.json` ที่นี่ก่อน index เข้า ChromaDB

```json
{
  "url": "http://10.192.133.33/smc/index.php?...",
  "title": "มาตรฐาน 5ส.",
  "text": "เนื้อหาบทความ...",
  "content_type": "general",
  "content_hash": "abc123...",
  "links": [{"text": "ลิงก์", "href": "http://..."}],
  "images": [{"url": "...", "alt": "...", "ocr_text": ""}],
  "processed_at": 1740000000.0
}
```

**ถ้าอยากดูว่าระบบเก็บข้อมูลอะไรจากบทความไหน:**
```bash
cat data/processed/10.192.133.33_smc_index.php_option_com_content*.json | python3 -m json.tool | head -50
```

---

## 6. `data/vectorstore/` — ChromaDB (Vector Database)

ระบบเก็บ embedding ของทุก chunk ไว้ในนี้ ใช้สำหรับ semantic search

| ไฟล์/โฟลเดอร์ | ความหมาย |
|---|---|
| `chroma.sqlite3` | ฐานข้อมูลหลัก (อย่าลบ!) |
| `*.bin` | Vector embedding ไฟล์ |

**ถ้าอยากดูข้อมูลใน collection:**
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
import yaml
from src.vectorstore.chroma import ChromaVectorStore
cfg = yaml.safe_load(open('configs/config.yaml'))
vs_cfg = cfg['vectorstore']
vs = ChromaVectorStore(
    persist_dir='data/vectorstore',
    collection_name=vs_cfg['collection_name'],
    embedding_model=vs_cfg['embedding_model'],
    bm25_path='data/bm25_index.json'
)
print(f'Total docs: {vs.collection.count()}')
results = vs.query('5S', top_k=2)
for r in results:
    print(r.metadata.get('url'), r.score)
"
```

---

## 7. `data/sessions/` — Session ผู้ใช้

บันทึกประวัติการสนทนาของแต่ละ user แยกตาม `session_id`

```json
{
  "session_id": "user-abc",
  "history": [
    {"role": "user", "content": "5S คืออะไร", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "last_intent": "GENERAL_QA",
  "context_url": "http://..."
}
```

**Session จะหมดอายุอัตโนมัติ** — ไม่ต้องดูแล แต่ถ้าอยากล้าง session ทั้งหมด:
```bash
rm data/sessions/*.json
```

---

## 8. `data/metrics.csv` — Log ทุก Request

```
timestamp,session_id,query,route,latency_ms,choices_len,sources_len,ans_len,...
2026-03-02 14:39:00,anon,5S คืออะไร,article_answer,8145.55,0,1,512,...
```

**วิเคราะห์ log:**
```bash
# คำถามที่ตอบช้าสุด 5 อันดับ
sort -t',' -k5 -rn data/metrics.csv | head -6

# route ที่ถูกใช้บ่อยสุด
cut -d',' -f4 data/metrics.csv | sort | uniq -c | sort -rn
```

---

## จะแก้/เพิ่ม Logic ของระบบ — แก้ตรงไหน?

| อยากแก้อะไร | แก้ที่ไฟล์ไหน |
|---|---|
| เปลี่ยน LLM / เพิ่ม prompt | `src/ai/safe_normalizer.py` |
| เปลี่ยนวิธีตัดสิน intent | `src/ai/router.py` |
| เพิ่มคำ alias อุปกรณ์ | `data/aliases.json` |
| เพิ่มกฎ routing พิเศษ | `src/core/chat_engine.py` |
| เปลี่ยน threshold confidence | `configs/config.yaml` → `rag.score_threshold` |
| แก้วิธีโชว์คำตอบ | `src/ai/response_generator.py` |
| เพิ่มผู้ติดต่อใหม่ | `data/records/*.json` |
| เพิ่มหน้าเว็บใหม่เข้าระบบ | `python scripts/sync_incremental.py --target-url "..."` |
| แก้วิธีดึงข้อมูลจากเว็บ | `src/ingest/fetch.py` และ `src/rag/article_cleaner.py` |

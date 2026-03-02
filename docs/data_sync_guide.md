# คู่มือการอัปเดตข้อมูล RAG System

> อัปเดตล่าสุด: 2026-03-02

เมื่อมีการเพิ่มหรือแก้ไขข้อมูลบนเว็บ SMC ให้ใช้ script ด้านล่างเพื่ออัปเดต Vector Database ของระบบ

---

## เตรียมความพร้อม

```bash
cd ~/Documents/NT/RAG/rag_web
source venv/bin/activate
```

> **หมายเหตุ:** ต้องอยู่ในเครือข่าย NT (ออฟฟิศ) ถึงจะเชื่อมเว็บ SMC ได้

---

## 1. อัปเดตเฉพาะหน้าที่เปลี่ยน (แนะนำ — ใช้บ่อยที่สุด)

### ขั้นตอน
```bash
# ขั้นที่ 1: ดูก่อนว่าจะมีอะไรเปลี่ยนบ้าง (ไม่แตะ DB)
python scripts/sync_incremental.py --dry-run

# ขั้นที่ 2: ถ้าโอเค รันจริง
python scripts/sync_incremental.py
```

### ผลลัพธ์ที่จะเห็น
| ข้อความ | ความหมาย |
|---|---|
| `[UPDATED]` | หน้านี้เนื้อหาเปลี่ยน → อัปเดต DB แล้ว |
| `[SKIP]` | หน้านี้เนื้อหาเดิม → ข้ามไม่ทำอะไร |
| `[ERROR]` | เข้าหน้านั้นไม่ได้ (timeout หรือ server ไม่ตอบ) |

### อัปเดตแค่หน้าเดียว
```bash
python scripts/sync_incremental.py \
  --target-url "http://10.192.133.33/smc/index.php?option=com_content&view=article&id=570"
```

---

## 2. Rebuild ฐานข้อมูลใหม่ทั้งหมด (ทำเดือนละครั้ง หรือเมื่อเว็บเปลี่ยนครั้งใหญ่)

Script นี้สร้างถังข้อมูลใหม่แยกต่างหาก ทดสอบก่อน แล้วค่อยสลับ — **ระบบหลักไม่หยุดทำงาน**

```bash
# ทำครบทุกขั้นตอนอัตโนมัติ (สร้าง → ทดสอบ → สลับ)
python scripts/swap_collection.py
```

### ถ้าอยากทำทีละขั้น
```bash
# ขั้นที่ 1: สร้างถังใหม่ (ยังไม่สลับ)
python scripts/swap_collection.py --build-only

# ขั้นที่ 2: ทดสอบถังใหม่
python scripts/swap_collection.py --test-only

# ขั้นที่ 3: สลับถัง (ต้อง restart API หลังจากนี้)
python scripts/swap_collection.py --swap-only
```

### Restart API หลังสลับถัง
```bash
pkill -f uvicorn
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## ถ้ามีปัญหา — วิธี Rollback

### กรณี 1: rollback config กลับถังเก่า
```bash
python scripts/swap_collection.py --rollback
# แล้ว restart API
```

### กรณี 2: rollback Vector DB กลับ backup
```bash
rm -rf data/vectorstore
cp -r data/vectorstore_backup_20260302 data/vectorstore
```

### กรณี 3: rollback โค้ดทั้งหมด (Git)
```bash
git checkout freeze-2026-03-02-pre-sync-test
```

---

## คำถามที่พบบ่อย

**Q: ถ้าเพิ่มหัวข้อใหม่บน SMC ระบบจะเจอมั้ย?**  
A: เจอ ถ้าหัวข้อนั้นถูกลิงก์จากหน้า index SMC  
ถ้าอยู่ลึกกว่านั้น ให้ระบุ URL ตรงๆ ด้วย `--target-url`

**Q: รันบ่อยแค่ไหน?**  
A: รัน `sync_incremental.py` ทุกครั้งที่มีการอัปเดตข้อมูลบน SMC  
รัน `swap_collection.py` ทุก 1-2 เดือน หรือเมื่อมีการเปลี่ยนแปลงครั้งใหญ่

**Q: รันแล้วระบบหลักพังมั้ย?**  
A: ไม่พัง — ทั้งสอง script แยกจากระบบหลัก API ที่รันอยู่ไม่กระทบครับ

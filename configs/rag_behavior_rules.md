# RAG System Behavior Guidelines

## Core Principle
**"ตอบให้ถูกตามข้อมูลจริงในระบบเท่านั้น"**

ห้าม: เดา, เติม, สรุปเกินข้อมูลที่มี

---

## Rule 1: Data Boundary Enforcement

**ถ้าข้อมูลไม่มีในระบบ → อธิบายว่า "ไม่มีข้อมูลในระบบปัจจุบัน" พร้อมเหตุผลเชิงโครงสร้าง**

### Implementation:
- Return routes: `contact_miss`, `team_miss`, `news_miss`, `knowledge_miss`
- Message format: "ไม่พบข้อมูล [X] ในระบบ เนื่องจาก [structural reason]"
- Structural reasons:
  - ไม่มีในดัชนี (not in index)
  - ไม่ตรงเงื่อนไข (doesn't match criteria)
  - อยู่นอกขอบเขตข้อมูล (outside data scope)

### Examples:
✅ "ไม่พบข้อมูลทีม 'ABC' ในระบบ เนื่องจากไม่มีในดัชนีทีมปัจจุบัน"
❌ "ทีม ABC น่าจะเป็น..." (hallucination)

---

## Rule 2: Ambiguity Clarification

**ถ้าคำถามกว้างหรือคลุมเครือ → เสนอเฉพาะตัวเลือกที่ระบบมีจริง**

### Implementation:
- Routes: `clarify`, `team_ambiguous`, `contact_ambiguous`
- Must provide ACTUAL candidates from index
- Suggestion format: numbered list with exact keys

### Examples:
✅ "พบหลายตัวเลือก:\n1) ผส.บลตน.\n2) ผส.พพ."
❌ "อาจหมายถึง X หรือ Y" (without index check)

---

## Rule 3: Structured Commands

**ถ้าคำถามเป็นคำสั่งระบบ (เช่น แสดงทั้งหมด, รายชื่อทั้งหมด) → ตอบแบบเชิงโครงสร้าง ไม่ใช้ fuzzy**

### Detection Keywords:
- "แสดงทั้งหมด", "รายชื่อทั้งหมด", "show all", "list all"
- "มีอะไรบ้าง", "มีทีมอะไรบ้าง", "what teams exist"

### Implementation:
- Bypass fuzzy matching
- Return ALL items from index (limit to reasonable size, e.g., 50)
- Format: structured list, not narrative

### Examples:
✅ "ทีมในระบบ:\n- HelpDesk\n- FTTx\n- Management SMC"
❌ Fuzzy match "ทั้งหมด" -> some team name

---

## Rule 4: Sensitive Content Protection

**ถ้าข้อมูลมาจากบทความที่มีข้อมูลอ่อนไหว (รหัสผ่าน / IP / ตารางภาพ) → ห้ามแสดงเนื้อหา**

### Detection:
- Keywords: "password", "รหัสผ่าน", "username", "user", "passwd", "credential", "login"
- Image-heavy articles (text < 400 chars, images >= 1)
- Tables with IP/credentials

### Implementation:
- Return link-only response
- Message: "บทความนี้มีข้อมูลอ่อนไหว กรุณาเปิดดูจากแหล่งที่มา\nแหล่งที่มา: [URL]"
- Route: `article_sensitive`

### Examples:
✅ "บทความนี้เนื้อหาหลักเป็นรูปภาพตาราง (มีข้อมูลผู้ใช้/รหัสผ่าน)..."
❌ Extracting table with passwords via OCR

---

## Rule 5: Index-Only Responses

**ห้ามตอบนอกเหนือ index ที่โหลดมา (KnowledgePack / TeamIndex / PositionIndex)**

### Implementation:
- All answers MUST trace back to:
  - `self.team_index` (for teams)
  - `self.position_index` (for positions/roles)
  - `self.knowledge_pack` (for facts)
  - `self.vs` (for articles via vector search)
  - `records` (for contacts)

### Validation:
- NO external API calls
- NO LLM-generated entities
- NO assumptions about data not in index

---

## Team Query Special Rules

### Use Case 1: Member Lookup
**Input**: "สมาชิกงาน HelpDesk"

**Process**:
1. Canonicalize: strip prefixes, brackets
2. Try alias match
3. Try exact match
4. Try fuzzy match (0.75 threshold)
5. Return members or ambiguous

---

### Use Case 2: List All Teams
**Input**: "แสดงทีมทั้งหมด", "รายชื่อทีม"

**Process**:
1. Detect "show all" intent
2. Return ALL keys from `self.team_index`
3. Format as simple list
4. NO fuzzy matching

**Implementation**:
```python
if "ทั้งหมด" in query or "รายชื่อ" in query:
    all_teams = list(self.team_index.keys())
    return f"ทีมในระบบ ({len(all_teams)} ทีม):\n" + "\n".join(f"- {t}" for t in all_teams)
```

---

### Use Case 3: Alias with No Real Team
**Input**: "สมาชิกงาน SMC"

**Process**:
1. Alias matched: "smc" -> ["management smc", ...]
2. No exact match found for any candidate
3. Fuzzy search finds close matches

**Response Options**:
- If 1 match (>=0.80): return that team
- If multiple close (within 0.03): return ambiguous
- If none: "ไม่มีทีม 'SMC' แยกในระบบ แต่พบทีมที่เกี่ยวข้อง: [suggestions]"

---

## Testing Requirements

Each rule must have regression test:
1. Data boundary: verify `_miss` routes return correct messages
2. Ambiguity: verify suggestions match index keys exactly
3. Structured commands: verify "show all" returns full index
4. Sensitive content: verify link-only responses for sensitive articles
5. Index-only: verify no hallucinated entities

---

## Prompt Integration

All LLM prompts must include:
- "Use ONLY information from provided context"
- "If answer not in context, say 'ไม่มีข้อมูล'"
- "Do NOT infer, guess, or extrapolate"
- "For sensitive content (passwords/IPs), return link only"

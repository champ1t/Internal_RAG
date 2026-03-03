
import sys
import time
sys.path.append('.')

from src.rag.article_cleaner import (
    clean_article_content, 
    extract_structured_facts, 
    format_fact_item, 
    has_structured_content,
    is_metadata_dominated
)

def test_fast_path_v2_formatting():
    print("Test 1: Fast-Path V2 Formatting")
    print("="*40)
    
    raw_content = """การจ่ายงานเลขหมายวงจรเช่า
1. การจ่ายงานเลขหมายวงจรเช่าในระบบ SCOMS ให้จ่ายงานไปยัง กองงานในพื้นที่ดังต่อไปนี้ สงขลา 09890X0026:ระบบสื่อสารข้อมูล (สงขลา <PHONE>) สุราษฎร์ธานี 09884X0999:สื่อสารข้อมูลสุราษฎร์ธานี
2. ให้ดำเนินการส่ง SMS แจ้งการจ่ายงานวงจรเช่าทุกครั้งเมื่อมีการจ่ายงานไปยังพื้นที่ โดยกำหนดแนวทางในการส่ง SMS ดังนี้
   a. รูปแบบข้อความ SMS
   c. URL สำหรับส่ง SMS : http://203.113.6.76/SMS_APP
   e. ผู้ส่ง/เบอร์ผู้ส่ง ให้ระบุเลขหมาย " <PHONE> "
แก้ไขล่าสุด ใน วันศุกร์ที่ 15 มิถุนายน 2018 เวลา 16:55 น.
Joomla 1.5 Templates by Joomlashack (http://www.joomlashack.com)"""

    print("Checking structure...")
    cleaned = clean_article_content(raw_content, keep_metadata=True)
    has_struct = has_structured_content(cleaned)
    print(f"has_structured_content: {has_struct}")
    
    print("\nExtracting facts...")
    facts = extract_structured_facts(cleaned)
    for i, f in enumerate(facts):
        print(f"--- Fact {i+1} ---")
        print(f)
        
    # Check for Joomla leakage
    if any("Joomla" in f for f in facts):
        print("\n❌ FAILED: Joomla footer leaked into facts")
    else:
        print("\n✅ SUCCESS: No Joomla footer")
        
    print("\n" + "="*40)

def test_metadata_guard():
    print("Test 2: Metadata Guard")
    print("="*40)
    
    meta_content = """ข่าวสารแก้ user สำหรับลูกค้า context private
เขียนโดย Maru^^
วันพฤหัสบดีที่ 06 กุมภาพันธ์ 2014 เวลา 17:42 น.
วันที่แก้ไขล่าสุด วันพฤหัสบดีที่ 06 กุมภาพันธ์ 2014 เวลา 18:11 น.
Joomla 1.5 Templates by Joomlashack (http://www.joomlashack.com)"""

    is_dominated = is_metadata_dominated(meta_content)
    print(f"Content:\n{meta_content}\n")
    print(f"is_metadata_dominated: {is_dominated}")
    
    if is_dominated:
        print("✅ SUCCESS: Detected metadata domination")
    else:
        print("❌ FAILED: Failed to detect metadata domination")

    print("\n" + "="*40)

def test_bras_ip_fastpath():
    print("Test 3: Bras IP Fast-Path (V2)")
    print("="*40)
    
    ip_content = """Bras IP
Surat-BRAS1 : 110.164.252.1
Surat-BRAS2 : 110.164.252.2
Phuket-BRAS1 : 110.164.253.1
Phuket-BRAS2 : 110.164.253.2
Songkhla-BRAS1 : 110.164.254.1
Songkhla-BRAS2 : 110.164.254.2
Yala-BRAS1 : 110.164.255.1
Yala-BRAS2 : 110.164.255.2
"""
    cleaned = clean_article_content(ip_content)
    has_struct = has_structured_content(cleaned)
    print(f"has_structured_content: {has_struct}")
    
    facts = extract_structured_facts(cleaned)
    print(f"Extracted {len(facts)} facts")
    for f in facts[:3]:
        print(f"- {f}")
        
    if has_struct and len(facts) >= 8:
         print("✅ SUCCESS: Detected IP table fast-path")
    else:
         print(f"❌ FAILED: struct={has_struct}, facts={len(facts)}")

if __name__ == "__main__":
    test_fast_path_v2_formatting()
    test_metadata_guard()
    test_bras_ip_fastpath()

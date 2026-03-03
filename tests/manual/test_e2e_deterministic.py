
import requests
import json
import time

def test_chat_deterministic():
    url = "http://localhost:8000/chat"
    
    test_cases = [
        {"q": "ระบุสมาชิก fttx", "expected": "รายชื่อสมาชิก"},
        {"q": "ขอตารางเบอร์ omc", "expected": "🔗 [เปิดไฟล์/หน้าเว็บ]"},
        {"q": "ใครคือผจ", "expected": "คุณต้องการค้นหาทีมใด"},
        {"q": "เบอร์ helpdesk", "expected": "<PHONE>"}
    ]
    
    print("=== FINAL E2E DETERMINISTIC TEST ===")
    # Load config from yaml
    import yaml
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    from src.core.chat_engine import ChatEngine
    engine = ChatEngine(cfg)
    
    for case in test_cases:
        print(f"User Query: {case['q']}")
        resp = engine.process(case['q'])
        print(f"Result Route: {resp.get('route')}")
        print(f"Answer: {resp.get('answer')[:100]}...")
        # Check for expected content
        if case['expected'] in resp.get('answer'):
            print("✅ SUCCESS")
        else:
            print("❌ FAILED (Check Output)")
        print("-" * 30)

if __name__ == "__main__":
    test_chat_deterministic()

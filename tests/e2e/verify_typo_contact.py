import sys
import os
import yaml
from src.core.chat_engine import ChatEngine

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/config.yaml")
    engine = ChatEngine(cfg)
    
    # Query with typo "บอร์" instead of "เบอร์"
    q = "บอร์ติดต่อคุณสมบูรณ์"
    print(f"\n[TEST] Query: '{q}'")
    
    res = engine.process(q)
    print(f"Route: {res.get('route')}")
    print(f"Answer: {res.get('answer')}")
    
    if "<PHONE>" in res.get("answer", ""):
        print("[RESULT] PASS")
    else:
        print("[RESULT] FAIL")

if __name__ == "__main__":
    main()

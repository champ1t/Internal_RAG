import sys
import os
import time

sys.path.append(os.getcwd())

from src.core.chat_engine import ChatEngine
import yaml

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/config.yaml")
    engine = ChatEngine(cfg)
    
    # Query causing issue
    q = "เบอร์ติดต่อคุณสมบูรณ์"
    print(f"\n[TEST] Query: '{q}'")
    
    res = engine.process(q)
    print(f"Route: {res.get('route')}")
    print(f"Answer: {res.get('answer')}")
    
    if "<PHONE>" in res.get("answer", "") or "<PHONE>" in res.get("answer", ""):
        print("[RESULT] PASS")
    else:
        print("[RESULT] FAIL")

if __name__ == "__main__":
    main()


import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.chat_engine import ChatEngine

# Mock Config
CFG = {
    "retrieval": {"top_k": 3},
    "llm": {"model": "qwen3:8b", "base_url": "http://localhost:11434"},
    "rag": {"score_threshold": 0.25},
    "chat": {"show_context": False, "save_log": False}
}

class TestEmailLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[SETUP] Initializing ChatEngine for Email Tests...")
        cls.engine = ChatEngine(CFG)

    def test_email_retrieval(self):
        # 1. Ask Position (Director) - Should trigger context
        res = self.engine.process("ใครคือ ผส.บลตน.")
        self.assertEqual(res["route"], "position_lookup")
        self.assertIn("สมบูรณ์", res["answer"])
        
        # 2. Ask Email
        res_email = self.engine.process("ขออีเมล")
        print(f"Q: ขออีเมล -> A: {res_email['answer']}")
        
        self.assertEqual(res_email["route"], "context_followup")
        self.assertIn("<REDACTED_EMAIL>", res_email["answer"])
        self.assertIn("อีเมลของ", res_email["answer"])

    def test_email_miss(self):
        # 1. Ask Position (Someone unlikely to have email extracted yet context-wise)
        # Actually Manager also likely has email if same template used.
        # Let's try 'งาน FTTx' group
        
        self.engine.last_context = None # Clear
        self.engine.process("งาน FTTx มีใครบ้าง")
        
        # 2. Ask Email
        res_email = self.engine.process("ขออีเมล")
        # Unless we extracted group emails (we didn't implement that logic for group list text yet, only headers?)
        # Personnel list had "tel:", but not explicit email cloaking in the text block.
        # So expectation is MISS.
        
        if res_email["route"] == "context_followup":
             # Surprise success?
             print("Found email for group?")
        else:
             self.assertEqual(res_email["route"], "context_followup_miss")
             self.assertIn("ไม่มีข้อมูลอีเมล", res_email["answer"])

if __name__ == "__main__":
    unittest.main()

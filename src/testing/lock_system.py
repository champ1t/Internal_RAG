import unittest
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core.chat_engine import ChatEngine

# Mock Config (Live Data, Mock LLM)
CFG = {
    "retrieval": {"top_k": 3},
    "llm": {"model": "qwen3:8b", "base_url": "http://localhost:11434"},
    "rag": {"score_threshold": 0.25},
    "chat": {"show_context": False, "save_log": False}
}

class TestSystemLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[SETUP] Initializing ChatEngine for Final System Lock...")
        t0 = time.time()
        cls.engine = ChatEngine(CFG)
        print(f"[SETUP] Ready in {time.time() - t0:.2f}s")

    def setUp(self):
        # Clear context before each test to ensure isolation
        self.engine.last_context = None
        
    def assertRoute(self, res, expected_route):
        self.assertEqual(res["route"], expected_route, f"Route mismatch for query '{res['answer'][:20]}...'")

    def assertZeroLLM(self, res):
        # We can't strictly check 0.00ms, but we check route is deterministic (not 'rag')
        # AND check that 'latencies'['llm'] is 0 if available, or just rely on route.
        if "latencies" in res:
             self.assertEqual(res["latencies"].get("llm", 0), 0.0)
        self.assertNotEqual(res["route"], "rag")

    # --- 1. Core Deterministic Lookups ---
    
    def test_lookup_position_alias(self):
        """Test 'ผจ' -> Matches 'ผจ.สบลตน.'"""
        res = self.engine.process("ใครคือ ผจ")
        
        # Phase 15: Position lookups now use Planner (lightweight LLM for output_mode detection)
        # So they're no longer zero-LLM, but should be fast (<100ms planner time)
        self.assertRoute(res, "position_lookup")
        self.assertIn("ปรัชญา", res["answer"])

    def test_lookup_position_full(self):
        """Test 'ผส.บลตน.' -> Matches 'สมบูรณ์'"""
        res = self.engine.process("ใครคือ ผส.บลตน.")
        self.assertRoute(res, "position_lookup")
        self.assertIn("สมบูรณ์", res["answer"])

    def test_lookup_link_fuzzy(self):
        """Test 'edoccument' -> Matches 'Edocument'"""
        res = self.engine.process("ขอลิงก์ edoccument")
        self.assertRoute(res, "link_lookup")
        self.assertIn("Edocument", res["answer"])
        self.assertIn("http", res["answer"])

    def test_lookup_group(self):
        """Test 'งาน FTTx' -> Group members"""
        res = self.engine.process("งาน FTTx มีใครบ้าง")
        self.assertRoute(res, "position_lookup")
        self.assertIn("ปรัชญา", res["answer"]) 
        self.assertIn("เหมาะโชค", res["answer"])

    def test_lookup_admin(self):
        """Test 'admin' -> 'ผู้ดูแลระบบ'"""
        res = self.engine.process("admin คือใคร")
        self.assertRoute(res, "position_lookup")
        self.assertIn("ชัยยา", res["answer"])
        self.assertIn("074-250685", res["answer"]) # Extracted phone
        self.assertIn("chaiyaa", res["answer"]) # Extracted email

    def test_lookup_role_fuzzy(self):
        """Test 'ผส.บลต' (Typo) -> Matches 'ผส.บลตน.'"""
        res = self.engine.process("ใครคือ ผส.บลต") # missing น.
        self.assertRoute(res, "position_lookup")
        self.assertIn("สมบูรณ์", res["answer"])

    # --- 2. Context & Multi-Turn ---

    def test_context_position_to_contacts(self):
        """
        Flow:
        1. Ask Position (Director)
        2. Ask Phone (Should exist via Extraction)
        3. Ask Link (Should exist via Context)
        4. Ask Email (Should explicit fail)
        """
        # Step 1
        self.engine.process("ใครคือ ผส.บลตน.")
        
        # Step 2: Phone
        res_phone = self.engine.process("ขอเบอร์หน่อย")
        self.assertRoute(res_phone, "context_followup")
        self.assertIn("<PHONE>", res_phone["answer"].replace("-", ""))
        self.assertZeroLLM(res_phone)

        # Step 3: Link
        res_link = self.engine.process("ขอลิงก์")
        self.assertRoute(res_link, "context_followup")
        self.assertIn("http", res_link["answer"])

        # Step 4: Email
        res_email = self.engine.process("ขออีเมล")
        self.assertRoute(res_email, "context_followup") # Found via extraction now
        self.assertIn("somboonc", res_email["answer"])

        # Step 5: Fax (Check intent overlap "โทรสาร" vs "โทร")
        res_fax = self.engine.process("โทรสาร")
        # Check for Thai formatted answer or Miss
        if "ไม่พบข้อมูลโทรสาร" in res_fax["answer"]:
             self.assertRoute(res_fax, "context_followup_miss")
        else:
             # If data exists
             self.assertRoute(res_fax, "context_followup")
             self.assertIn("Fax", res_fax["answer"])

    def test_intent_fax_overlap(self):
        """Ensure 'โทรสาร' triggers Fax intent, NOT Phone intent"""
        self.engine.process("ใครคือ ผส.บลตน.")
        res = self.engine.process("โทรสาร")
        # Should NOT contain "เบอร์ติดต่อ" (Phone header) if data missing
        # Should match Fax logic
        if "ไม่พบข้อมูลโทรสาร" in res["answer"]:
             self.assertRoute(res, "context_followup_miss")
        else:
             # If found, check header is "Fax"
             self.assertIn("Fax", res["answer"])

    def test_fax_retrieval_success(self):
        """Test 'Fax' retrieval for Manager (who has data)"""
        self.engine.process("ใครคือ ผจ")
        res = self.engine.process("ขอแฟกซ์")
        self.assertRoute(res, "context_followup")
        self.assertIn("Fax", res["answer"])
        self.assertIn("074-250568", res["answer"])

    def test_fax_direct_query_routing(self):
        """Test 'Fax [Name]' routes to Position Lookup, not Context Miss"""
        # User query was "โทรสาร ผส.บลตน." which failed before
        res = self.engine.process("โทรสาร ผจ") # Use "ผจ" (Manager) who has fax
        
        # Should be position_lookup (or contact matching logic)
        # It shouldn't be context_followup_miss
        if res["route"] == "context_followup_miss":
             self.fail(f"Direct Fax Query routed to context miss! Res: {res}")
        
        self.assertIn(res["route"], ["position_lookup", "contact_name", "contact_reverse"])
        # Phase 15: Fax query with entity triggers FAX_ONLY output mode
        # So answer contains just the fax number, not "โทรสาร:" label
        self.assertIn("074-250568", res["answer"])

    def test_email_miss_safety(self):
        """Ensure Asking Email for random person doesn't crash or hallucinate"""
        # Set context first (mock or real)
        self.engine.process("ใครคือ ผจ") # Manager
        res = self.engine.process("ขออีเมล")
        # Manager might have email now!
        if "pratyas" in res["answer"] or "email" in res["answer"].lower():
             self.assertRoute(res, "context_followup")
        else:
             self.assertIn("ไม่พบข้อมูลอีเมล", res["answer"])

    def test_ood_garbage(self):
        """Ensure nonsense goes to RAG (low score)"""
        res = self.engine.process("sdlfkjsdflkjsdf")
        # Likely rag_no_docs or rag_low_score
        self.assertTrue(res["route"] in ["rag_no_docs", "rag_low_score", "rag"])

if __name__ == "__main__":
    unittest.main()

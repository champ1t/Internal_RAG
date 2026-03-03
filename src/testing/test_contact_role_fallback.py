
import unittest
from src.rag.handlers.contact_handler import ContactHandler
from src.rag.handlers.directory_handler import DirectoryHandler

class TestContactRoleFallback(unittest.TestCase):
    def setUp(self):
        # Mock Position Index
        self.mock_index = {
            "ผส.บลตน.": [
                {
                    "name": "นายสมชาย ใจดี", 
                    "role": "ผส.บลตน.", 
                    "phones": ["<PHONE>", "<PHONE>"],
                    "emails": ["somchai@nt.com"]
                }
            ],
            "ผจ.สบลตน.": [
                 {
                    "name": "นางสาวสวย จัง", 
                    "role": "ผจ.สบลตน.", 
                    "phones": ["074-250685"],
                    "emails": []
                }
            ]
        }
        self.mock_records = [] # Empty contact book to force fallback
        self.directory = DirectoryHandler(self.mock_index, self.mock_records)

    def test_fallback_exact_match(self):
        # Query: "เบอร์ ผส.บลตน" -> Fallback to "ผส.บลตน"
        q = "เบอร์ ผส.บลตน."
        res = ContactHandler.handle(q, self.mock_records, directory_handler=self.directory)
        
        print(f"\n[Test Exact] Query: {q}")
        print(f"Answer: {res['answer']}")
        
        self.assertEqual(res["route"], "contact_role_fallback")
        self.assertIn("นายสมชาย", res["answer"])
        self.assertIn("<PHONE>", res["answer"])
        self.assertIn("ผส.บลตน.", res["answer"])

    def test_fallback_alias_fuzzy(self):
        # Query: "ขอเบอร์โทร ผจ สบลตน" (No dots, extra words)
        q = "ขอเบอร์โทร ผจ สบลตน"
        res = ContactHandler.handle(q, self.mock_records, directory_handler=self.directory)
        
        print(f"\n[Test Fuzzy] Query: {q}")
        print(f"Answer: {res['answer']}")
        
        # Should match "ผจ.สบลตน." via normalization (NoSpace/Scan)
        self.assertEqual(res["route"], "contact_role_fallback")
        self.assertIn("นางสาวสวย", res["answer"])
        self.assertIn("074-250685", res["answer"])

    def test_ambiguous_short_role(self):
        # Query: "เบอร์ ผส" -> Should fail or return many?
        # In this mock, we only have one "ผส..." so it might match if fuzzy scan allows partial
        # But let's see. 
        # DirectoryHandler._scan_matches checks substring.
        pass

if __name__ == "__main__":
    unittest.main()

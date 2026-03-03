import unittest
from src.directory.extract_positions import merge_entities

class TestProvenanceContracts(unittest.TestCase):
    
    def test_contract_1_canonical_merge(self):
        """
        Contract 1: Phone numbers with formatting differences MUST merge into a single canonical entry.
        0-7425-0685 == <PHONE>
        """
        data = [
            {"role": "Staff", "name": "User A", "phones": ["0-7425-0685"], "emails": [], "faxes": [], "source": "FileA"},
            {"role": "Staff", "name": "User A", "phones": ["<PHONE>"], "emails": [], "faxes": [], "source": "FileB"},
        ]
        
        results = merge_entities(data)
        self.assertEqual(len(results), 1, "Should merge to one entity")
        rec = results[0]
        
        # CURRENTLY FAILS: The system treats them as distinct keys.
        # Expectation: 1 phone number in list.
        self.assertEqual(len(rec["phones"]), 1, f"Expected 1 canonical phone, got: {rec['phones']}")
        
        # Verify sources combined
        # Key might be normalized (e.g. <PHONE>)
        # accessing by values() is safer if we don't know exact calc format yet
        all_sources = []
        for srcs in rec["phone_sources"].values():
            all_sources.extend(srcs)
            
        self.assertIn("FileA", all_sources)
        self.assertIn("FileB", all_sources)

    def test_contract_2_conflict_provenance(self):
        """
        Contract 2: Different phone numbers merge into list, preserving distinct sources.
        """
        data = [
            {"role": "Staff", "name": "User B", "phones": ["1111"], "emails": [], "faxes": [], "source": "FileA"},
            {"role": "Staff", "name": "User B", "phones": ["2222"], "emails": [], "faxes": [], "source": "FileB"},
        ]
        results = merge_entities(data)
        rec = results[0]
        
        self.assertEqual(len(rec["phones"]), 2)
        self.assertEqual(sorted(rec["phone_sources"]["1111"]), ["FileA"])
        self.assertEqual(sorted(rec["phone_sources"]["2222"]), ["FileB"])

    def test_contract_3_deduplication(self):
        """
        Contract 3: Identical data from same source merges idempotently.
        """
        data = [
             {"role": "Staff", "name": "User C", "phones": ["3333"], "emails": [], "faxes": [], "source": "FileA"},
             {"role": "Staff", "name": "User C", "phones": ["3333"], "emails": [], "faxes": [], "source": "FileA"},
        ]
        results = merge_entities(data)
        rec = results[0]
        self.assertEqual(len(rec["phones"]), 1)
        self.assertEqual(sorted(rec["phone_sources"]["3333"]), ["FileA"]) # Just one source instance (set logic)

if __name__ == "__main__":
    unittest.main()

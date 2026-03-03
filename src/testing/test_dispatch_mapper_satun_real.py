
import unittest
from src.rag.handlers.dispatch_mapper import DispatchMapper

class TestDispatchMapperSatunReal(unittest.TestCase):
    def test_satun_extraction(self):
        # Text provided by user
        article_text = """
การจ่ายงานเลขหมายวงจรเช่า
1. สตูล
   กองงานรวมสตูล
   
   สตูล วงจรปกติ
   09891Z0300:สตูล 300 ศูนย์ระบบสื่อสาร (<PHONE>,<PHONE>)

   สตูล วงจรบนเกาะ,Wi-net
   สตูล 400 ศูนย์สือสารไร้สาย

2. ระเบียบการส่ง SMS
   ให้ดำเนินการ...
"""
        data_map = DispatchMapper._parse_dispatch_article(article_text)
        
        if "สตูล" in data_map:
            content = data_map["สตูล"]
            print(f"DEBUG Satun Real Content:\n{content}")
            
            # Must capture the Code line
            self.assertIn("09891Z0300", content)
            
            # Must capture the "Wi-net" subheader or at least the line after it
            self.assertIn("สตูล 400", content)
            self.assertIn("ศูนย์สือสารไร้สาย", content) # Typo in source must be captured
            
            # Must NOT capture Policy "2. ระเบียบ"
            self.assertNotIn("ระเบียบการส่ง SMS", content)
            
        else:
            self.fail("Satun not found in map")

if __name__ == "__main__":
    unittest.main()

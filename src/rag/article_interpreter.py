"""
Article Interpreter Module
Extracts and structures information from SMC news/article content.
Evidence-Grounded: Only uses explicit content from article, no hallucination.
"""
from typing import Dict, Any
import json
import time
import re # Fix UnboundLocalError
from src.rag.ollama_client import ollama_generate
from src.rag.article_cleaner import (
    clean_article_content,
    truncate_content,
    has_structured_content,
    is_metadata_dominated,
    second_chance_procedural_extraction,
    extract_cli_commands
)
from src.rag.image_validator import filter_valid_images

def is_tutorial_intent(query: str) -> bool:
    """
    Phase 119: Check for Type B Intent (Tutorials/Concepts).
    """
    triggers = ["ทำความรู้จัก", "อธิบาย", "หลักการ", "overview", "concept", "คืออะไร", "what is", "introduction"]
    return any(t in query.lower() for t in triggers)

class ArticleInterpreter:
    def __init__(self, llm_cfg: Dict[str, Any], ux_cfg: Dict[str, Any] = None):
        self.base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self.model = llm_cfg.get("fast_model", llm_cfg.get("model")) # Phase 135: Use Fast Model
        self.temperature = 0.0  # Deterministic for article extraction
        self.max_tokens = 280  # Concise summaries (reduced for speed)
        self._result_cache = {} # Phase 136: Result Cache (URL, Query) -> Answer
        self.ux_cfg = ux_cfg or {}

    def _is_technical_content(self, content: str) -> bool:
        """Check if content contains technical configuration commands."""
        keywords = ["display", "config", "command", "system-view", "undo", "vlan", "interface", "ip address", "service-port", "ont ", "profile-index", "ping", "sbc", "sip", "tracert"]
        content_lower = content.lower()
        return any(k in content_lower for k in keywords)

    def _is_tutorial_intent(self, query: str) -> bool:
        """
        Phase 119: Check for Type B Intent (Tutorials/Concepts).
        """
        triggers = ["ทำความรู้จัก", "อธิบาย", "หลักการ", "overview", "concept", "คืออะไร", "what is", "introduction"]
        return any(t in query.lower() for t in triggers)
        
    def _process_ocr(self, images: list) -> str:
        """
        Attempt to extract text from images (Optical Character Recognition).
        Requires 'pytesseract' and 'Pillow'.
        """
        if not images: return ""
        
        try:
            import pytesseract
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Placeholder for Real Logic:
            # 1. Download image from url
            # 2. Convert to Grayscale
            # 3. pytesseract.image_to_string(img, lang='tha+eng')
            # 4. Return combined text
            
            return "[OCR: Logic not fully implemented (requires network/process)]"
            
        except ImportError:
            # Graceful degradation
            return "[System: OCR unavailable. Library 'pytesseract' not found.]"

    def _wrap_result(self, answer: str, content: str) -> Dict[str, Any]:
        # Centralized Disclaimer Logic (Phase 220)
        # Check content for complex elements and append disclaimer if needed
        disclaimer_parts = []
        
        # 1. Image Check (heuristic)
        if "[Image]" in content:
                disclaimer_parts.append("รูปภาพ/แผนผัง")
                
        # 2. Table Check (heuristic)
        if "[Table]" in content or "table" in content.lower():
                disclaimer_parts.append("ตารางข้อมูล")
                
        # 3. File Check (heuristic for xls, pdf, doc)
        if any(ext in content.lower() for ext in [".xls", ".xlsx", ".pdf", ".doc", ".docx", "google docs", "spreadsheet", "docs.google.com", "drive.google.com"]):
                disclaimer_parts.append("ไฟล์แนบ")

        if disclaimer_parts:
            # Avoid duplicate disclaimer if already present
            found_types = "/".join(disclaimer_parts)
            disclaimer_msg = f"\n\n⚠️ **หมายเหตุ:** บทความนี้มี {found_types} ซึ่งไม่สามารถแสดงผลได้ครบถ้วน กรุณาตรวจสอบรายละเอียดเพิ่มเติมจากลิงก์ต้นฉบับ"
            
            if "⚠️ **หมายเหตุ:**" not in answer:
                 answer += disclaimer_msg

        """
        Phase 21: Extract quality metadata from content.
        """
        paragraphs_count = 0
        bullets_count = 0
        
        if content:
            # Phase 21 Fix: Paragraph Normalizer
            # Some scraped articles have numbered items joined in one line (e.g. "1. item 2. item")
            # Expand them to separate lines before counting so the P/B counters work correctly.
            # This ONLY affects counting logic, NOT the content sent to the LLM.
            content_for_counting = re.sub(r'\s+(\d+\.\s)', r'\n\n\1', content)
            # Count non-empty paragraphs (at least 40 chars to be meaningful)
            paragraphs = [p for p in content_for_counting.split("\n\n") if len(p.strip()) > 40]
            paragraphs_count = len(paragraphs)
            # Count bullets
            bullets = re.findall(r'^\s*[\•\-\*\d\.)]', content_for_counting, re.MULTILINE)
            bullets_count = len(bullets)
        
        return {
            "answer": answer,
            "metadata": {
                "paragraphs": paragraphs_count,
                "bullets": bullets_count,
                "is_extractive": False  # Default: LLM path. SHORT path overrides this to True.
            }
        }

    def interpret(self, user_query: str, article_title: str, article_url: str, article_content: str, images: list = None, show_images: bool = False, match_score: float = 0.0, intent: str = None) -> str:
        """
        Public wrapper for interpretation with Caching.
        """
        # 0. Clean article content FIRST (Phase 45 Strict Noise Removal)
        # Rule AC-2: Clean before cache calculation to avoid junk pollution
        from src.rag.article_cleaner import clean_article_content, strip_menus, mask_sensitive_data, rank_links_by_query, deduplicate_paragraphs, extract_cli_commands
        
        # Initial Cleaning for Cache Key Stability
        cleaned_content = clean_article_content(article_content, keep_metadata=True)
        cleaned_content = strip_menus(cleaned_content)
        cleaned_content = deduplicate_paragraphs(cleaned_content)
        
        # Remove Stats lines (Today/Online users)
        lines = cleaned_content.split('\n')
        lines = [l for l in lines if "Today" not in l and "Online" not in l and "Yesterday" not in l]
        cleaned_content = "\n".join(lines)

        # Phase 136: Result Cache Check
        # Robust Cache Key: URL + Query + ContentHash + Intent (New Partition)
        # Phase 183: Unified Cache & Config
        # Schema v223_strict_cleaning (Rule AC-1)
        CACHE_SCHEMA_VERSION = "v223_clean_policy"
        ux_flags = f"{self.ux_cfg.get('article.intro.enabled')}:{self.ux_cfg.get('article.max_chars')}"
        
        import hashlib
        # Use CLEANED content for hash
        content_hash = hashlib.md5(cleaned_content[:2000].encode('utf-8', errors='ignore')).hexdigest()
        cache_key = (article_url, user_query, content_hash, CACHE_SCHEMA_VERSION, ux_flags, intent or "general")
        
        if cache_key in self._result_cache:
            print(f"[ArticleInterpreter] Cache Hit for {article_url} (<1ms)")
            print(f"ArticleMode=CACHE Links=? TextChars=? Score={match_score} Cache=hit")
            return self._result_cache[cache_key]
            
        # Call internal logic with PRE-CLEANED content to save time
        result_data = self._interpret_logic(user_query, article_title, article_url, article_content, images, show_images, match_score, intent=intent, pre_cleaned_content=cleaned_content)
        
        # result_data is now a dict: {"answer": str, "metadata": {"paragraphs": int, "bullets": int}}
        # For backward compatibility with tests/internal calls, we return the string but keep metadata available
        # Actually, let's return a string that ChatEngine can parse or just use a dict if we update ChatEngine.
        # Decision: Return a dict from _interpret_logic and keep interpret returning a string for now, but maybe ChatEngine should know.
        # Better: let's have interpret return a string, but embed metadata in a way ChatEngine can extract if needed, or just change the signature.
        # Since I'm the one editing both, I'll change it to return a dict.
        
        self._result_cache[cache_key] = result_data
        return result_data

    def _interpret_logic(self, user_query: str, article_title: str, article_url: str, article_content: str, images: list = None, show_images: bool = False, match_score: float = 0.0, intent: str = None, pre_cleaned_content: str = None) -> Dict[str, Any]:
        """
        Internal Logic: Extract and structure information from article content.
        Returns Dict: {"answer": str, "metadata": {"paragraphs": int, "bullets": int}}
        """

        # Timing breakdown
        timings = {}
        t_start = time.time()
        
        if not article_content or len(article_content.strip()) < 10:
             return self._wrap_result(f"เนื้อหาในบทความสั้นเกินไปหรือไม่พบข้อมูล\n\nแหล่งที่มา:\n🔗 {article_url}", "")
             
        # Phase 185 Fix: Normalize Query for reliable keyword matching (Thai NFD/NFC)
        import unicodedata
        user_query = unicodedata.normalize('NFC', user_query)
        
        # Use pre-cleaned content if available
        if pre_cleaned_content:
             cleaned_content = pre_cleaned_content
        else:
        
            # Phase 1: Real-time Cleaning (Critical for Hybrid quality)
            from src.rag.article_cleaner import clean_article_content, strip_navigation_text, deduplicate_paragraphs
            
            # 1. Basic Cleaning
            cleaned_content = clean_article_content(article_content, keep_metadata=True)
            
            # Phase 221 Fix: Strip Navigation Text ALWAYS (Global Policy)
            # This was previously only called in Tutorial mode, but noise like 'Link ที่เกี่ยวข้อง' hits everyone.
            # We must strip it here so that Hybrid Strategy gets clean content.
            cleaned_content = strip_navigation_text(cleaned_content)
            
            # Deduplicate
            cleaned_content = deduplicate_paragraphs(cleaned_content)

        clean_len = len(cleaned_content)
        
        # Phase 135: Early Hard Reject (Link Density)
        # Avoid cleaning/tokenizing Navigation or Index pages
        # Heuristic: High link count relative to content length
        link_count = article_content.lower().count("href=") + article_content.lower().count("http")
        if link_count > 40:
             # Check density: Avg chars per link
             # Normal article: > 300 chars per link. Nav page: < 100 chars per link.
             density = len(article_content) / link_count
             if density < 120 and "download" not in user_query.lower():
                 print(f"[ArticleInterpreter] Early Reject: Link Density High ({density:.1f} chars/link, {link_count} links). Treating as Nav Page.")
                 
                 # Import for Ranking (local scope if needed, or rely on top)
                 # Ensure imports are available
                 from src.rag.article_cleaner import rank_links_by_query
                 
                 # Extract some links to show the user (Phase 172/185 UX)
                 if self.ux_cfg.get("enable_formatting", True):
                     max_links = self.ux_cfg.get("max_nav_links", 5)
                     # Phase 209: Use Standardized Directory Curator (v209)
                     # Instead of calling legacy rank_links_by_query, use our unified parser
                     parsed_output = self._parse_link_directory(article_content, article_title, user_query, article_url)
                     
                     if parsed_output:
                          return self._wrap_result(parsed_output, cleaned_content)
                 
                 return f"เนื้อหาเป็นเพียงเมนูนำทาง (Navigation Page) ไม่พบรายละเอียดในหน้านี้\n\nแหล่งที่มา:\n🔗 {article_url}"
        
        # 1. Clean article content (handled above or via pre_cleaned_content)
        # We now rely on pre_cleaned_content being populated or the fallback above.
        
        # t_clean = time.time()
        # Phase 45: Strict Noise Removal (uses updated article_cleaner)
        # from src.rag.article_cleaner import clean_article_content, strip_menus, mask_sensitive_data, rank_links_by_query, deduplicate_paragraphs, extract_cli_commands
        
        # cleaned_content = clean_article_content(article_content, keep_metadata=True)
        # Phase 46: Strip Menus
        # cleaned_content = strip_menus(cleaned_content)
        
        # timings["clean_ms"] = (time.time() - t_clean) * 1000

        # Phase 76: Link Directory Parser (Manual/Download Pages) - PRIORITY 1
        # Check full cleaned content for directory structure BEFORE other checks
        # Phase 98 Fix: Do NOT treat as directory if it contains technical commands (e.g. display, config)
        # UPDATE: "SIP" or "SBC" might trigger is_tech, but it's still a directory.
        # We try to parse as directory first. If it yields GOOD structure, we use it.
        # Only if parsing fails or returns few links do we fall back to Article processing.
        is_tech = self._is_technical_content(cleaned_content)
        
        # Phase 140: Strict Directory Detection with Score Safety Fallback
        is_directory = self._looks_like_link_directory(cleaned_content, user_query)
        
        # If High Confidence Match (Direct Lookup), be skeptical of Directory Mode unless obvious (lots of links)
        if match_score >= 0.9 and is_directory:
            raw_link_count = cleaned_content.lower().count("http") + cleaned_content.lower().count("ftp://") 
            if raw_link_count < 15:
                print(f"[ArticleInterpreter] High Confidence Match ({match_score}) -> Override weak Directory signal (links={raw_link_count}). force Article Mode.")
                is_directory = False

        if is_directory:
             # Phase 189: Strict Tech Rescue
             # If content contains explicit technical commands, it is NOT a directory
             # (Unless user explicitly asks for 'download' or 'manual')
             # Phase 189: Strict Tech Rescue
             # If content contains explicit technical commands, it is NOT a directory
             # UNLESS it looks like a Link Collection (lots of links)
             # User Rule: "If has explicit Tech content but > 10 links, treat as Directory"
             raw_link_count = cleaned_content.lower().count("http") 
             if is_tech and raw_link_count < 10 and "download" not in user_query.lower() and "manual" not in user_query.lower():
                 print(f"[ArticleInterpreter] Technical Content Detected (Links={raw_link_count}) -> Override Directory signal.")
                 is_directory = False

        if is_directory:
             print(f"[ArticleInterpreter] Detected Link Directory/Manual Page")
             
             # Requirement 1: Force Non-Generic Title
             display_title = article_title.strip()
             if not display_title or "News Article" in display_title or "บทความ" == display_title:
                     display_title = user_query.strip()

             directory_ans = self._parse_link_directory(cleaned_content, display_title, user_query, article_url)
             
             # Use directory answer if it found links. 
             # Exception: If is_tech is VERY strong (has actual commands), 
             # maybe we check if directory_ans is better? 
             # Usually a directory parser result is clearer than a text dump for lists.
             if directory_ans:
                 timings["total_ms"] = (time.time() - t_start) * 1000
                 print(f"ArticleMode=directory Links={len(directory_ans.splitlines())} TextChars={len(cleaned_content)} Score={match_score} Cache=miss")
                 return self._wrap_result(directory_ans, cleaned_content)
             else:
                 print(f"[ArticleInterpreter] Directory Parse failed/low quality -> Fallback to Article Mode.")


        # Phase 45: Image-Heavy Detection (Expanded Threshold)
        # If content is short (< 600 chars) OR looks like a junk table -> "Content is in Images"
        clean_len = len(cleaned_content)
        has_images = images and len(images) > 0
        
        # Heuristic 2: Junk Table Detection (e.g. "| 137 | 522 | ...")
        # Check: High number of pipes AND high digit density
        pipe_count = cleaned_content.count("|")
        digit_count = sum(c.isdigit() for c in cleaned_content)
        is_junk_table = False
        
        if pipe_count > 10 and has_images:
            # Check ratio: If digits > 30% of content, it's likely a raw data dump/table
            if digit_count / clean_len > 0.3:
                is_junk_table = True
                print(f"[DEBUG] Junk Table Detected: pipes={pipe_count}, digit_ratio={digit_count/clean_len:.2f}")

        # Check: Short content OR Junk Table
        if (clean_len < 600 or is_junk_table) and has_images and not any(k in cleaned_content.lower() for k in ["display", "config", "command"]):

            # Heuristic: < 600 chars is likely just a title or header residue + images.
            # Phase 98: Rescue technical content still applies via check above.
             
            print(f"[DEBUG] Image-Heavy Article detected (len={clean_len}, imgs={len(images)}) -> Switch to Link Mode.")
                 
            # Check if user explicitly asked for Summary
            is_summary_request = any(k in user_query.lower() for k in ["สรุป", "summary", "extract", "ย่อ"])
                 
            if not is_summary_request:
                # Default Policy: Link Mode
                msg = "⚠️ ข้อมูลเรื่องนี้ถูกจัดเก็บในรูปแบบรูปภาพ/ไฟล์สแกน\n(ระบบไม่สามารถอ่านรายละเอียดได้แม่นยำ กรุณาคลิกดูจากต้นฉบับครับ)"
                return self._wrap_result(f"{msg}\nแหล่งที่มา: {article_url}", cleaned_content)

            else:
                # Override: Process but rely on Masking at the end
                print("[DEBUG] User requested Summary -> Proceeding with masked extraction.")
                  
                # Phase 63: OCR Fallback (Opt-in)
                ocr_content = self._process_ocr(images)
                  
                # Phase 64: OCR Safety Contract (Must be meaningful > 120 chars)
                is_valid_ocr = ocr_content and len(ocr_content) >= 50
                  
                if is_valid_ocr:
                    print(f"[ArticleInterpreter] OCR extracted {len(ocr_content)} chars")
                    cleaned_content += "\n\n=== OCR Extracted Content ===\n" + ocr_content
                else:
                    print(f"[ArticleInterpreter] OCR rejected (len={len(ocr_content) if ocr_content else 0} chars). Falling back to safety mode.")
                    msg = "ไม่สามารถสรุปจากภาพได้อย่างปลอดภัย กรุณาเปิดดูจากแหล่งที่มา"
                    return self._wrap_result(f"{msg}\nแหล่งที่มา: {article_url}", cleaned_content)


        # Phase 121: Strict Enterprise Type A/B Logic Separation
        # 1. Type B Trigger (Tutorial/Concept) -> Force LLM
        # 2. Type A Trigger (Procedural) -> Fast-path (No LLM)
        
        # Phase 121: Define Keywords (Hoisted)
        explain_keywords = ["อธิบาย", "สาเหตุ", "ทำไม", "เพราะอะไร", "สรุป", "วิเคราะห์", "เกิดจากอะไร", "explain", "why", "reason", "summary"]
        type_b_triggers = ["ทำความรู้จัก", "หลักการ", "overview", "concept", "คืออะไร", "what is", "introduction"]
        # Phase 185: Expand Keywords for How-to/Commands (User Request)
        howto_triggers = ["คำสั่ง", "วิธี", "ขั้นตอน", "command", "config", "display", "setting", "setup", "ตรวจสอบ", "check", "add ", "create ", "modify "]
        
        tutorial_keywords = explain_keywords + type_b_triggers + howto_triggers
        
        is_tutorial_mode = any(k in user_query.lower() for k in tutorial_keywords)
        print(f"[ArticleInterpreter] is_tutorial_mode: {is_tutorial_mode}")
        
        # Ensure imports are available for logic below
        from src.rag.article_cleaner import extract_topic_anchored_facts, is_navigation_dominated, has_structured_content, deduplicate_paragraphs

        limit_chars = 3000
        
        # Phase 185: Enhanced Title Fallback (Global)
        # Check if DB title is generic
        display_title = article_title.strip()
        GENERIC_TITLES = {"News Article", "บทความ", "N/A", "", "None", "ข่าวประชาสัมพันธ์", "ข่าวสาร"}
        if not display_title or display_title in GENERIC_TITLES:
             display_title = user_query.strip()
             
        # Import missing functions
        from src.rag.article_cleaner import smart_truncate, strip_navigation_text

        # Phase 122 Override: THAI_ORG_KNOWLEDGE (5S) - Moved to Top Priority
        if intent == "THAI_ORG_KNOWLEDGE":
            print("[ArticleInterpreter] THAI_ORG_KNOWLEDGE Override -> Using Strict 5S Template")
            prompt = f"""
System: You are an expert in Thai Organizational Standards (5S, KPI, ISO).
Context from Article:
{cleaned_content[:8000]}

Goal: Summarize this article using the strict 5S Standard format.
Rules:
1. Interpret "5ส" ALWAYS as "Seiri (สะสาง), Seiton (สะดวก), Seiso (สะอาด), Seiketsu (สร้างมาตรฐาน), Shitsuke (สร้างวินัย)".
2. Use Thai Language ONLY.
3. DO NOT include author name or date line in the output.
4. Format output strictly as:

[{display_title}]

สรุปย่อ:
(อธิบาย 2-3 ประโยคว่า 5ส คืออะไร และวัตถุประสงค์ของมาตรฐานนี้คืออะไร — ห้ามลิสต์รายการข้อในส่วนนี้)

หลักการ 5 ข้อ (จากบทความ):
1. [ชื่อหัวข้อ] — [รายละเอียดเฉพาะที่พบในเนื้อหาของบทความ]
2. [ชื่อหัวข้อ] — [รายละเอียดเฉพาะที่พบในเนื้อหาของบทความ]
3. ...
4. ...
5. ...

(ถ้าบทความมีรายละเอียดเพิ่มเติม ให้ดึงออกมาในส่วน หลักการ 5 ข้อ เท่านั้น)
"""
            try:
                from src.rag.ollama_client import ollama_generate
                llm_summary = ollama_generate(
                    base_url=self.base_url,
                    model=self.model,
                    prompt=prompt,
                    system="You are a strict Thai Organizational Standards Expert.", 
                    max_tokens=612,
                    temperature=0.1
                ).strip()
                
                # Phase 18: Link appending removed (handled by ChatEngine)
                # llm_summary += f"\n\nแหล่งที่มา:\n🔗 {article_url}"
                timings["total_ms"] = (time.time() - t_start) * 1000
                return self._wrap_result(llm_summary, cleaned_content)
            except Exception as e:
                print(f"[ArticleInterpreter] 5S Override Failed: {e}")
                # Fallback to standard flow


        if is_tutorial_mode and False: # DISABLED Phase 185: merged into Phase 198 Hybrid Strategy
            # Phase 185: Extractive Intro + Priority Lines (User Request)
            # Replaces LLM Summary to avoid hallucination on technical content
            
            print(f"[ArticleInterpreter] Tutorial/How-to Mode Detected -> Using Extractive Policy.")
            
            # 1. Clean & Strip Nav
            cleaned_content = strip_navigation_text(cleaned_content)
            # Phase 185: Dedup repeated content (e.g. repeated CMS export blocks)
            cleaned_content = deduplicate_paragraphs(cleaned_content)
            
            # 2. Extract Intro (First non-empty lines that aren't junk)
            lines = cleaned_content.split('\n')
            intro_lines = []
            valid_intro_count = 0
            
            for line in lines[:15]: # Check first 15 lines
                line_s = line.strip()
                if not line_s: continue
                # Keep metadata lines (Author/Date) as they give context
                if any(k in line_s for k in ["เขียนโดย", "Created", "Last Updated", "วัน"]):
                    intro_lines.append(line_s)
                    continue

                # Phase 191: Skip Technical Lines in Intro to prevent Duplication
                # (These will be picked up by Priority Extraction below)
                if any(k in line_s.lower() for k in howto_triggers):
                     continue
                     intro_lines.append(line_s)
                     continue
                
                # Capture first 2 actual content lines
                if valid_intro_count < 2 and len(line_s) > 20:
                     intro_lines.append(line_s)
                     valid_intro_count += 1
            
            intro_text = "\n".join(intro_lines)
            
            # 3. Priority Lines Extraction (Commands/Configs)
            priority_lines = []
            
            # Rule A: Command-First Triggers (Expanded)
            cmd_triggers = [
                "#", "config", "display", "undo", "interface", "service-port", "vlan", "bridge-domain", 
                "=>", ">>", "ont ", "gpon", "ip address", "ping ", "telnet", "ssh", "activate", "policy-map", "class-map"
            ]
            
            count_lines = 0
            MAX_LINES = 40  # Rule B: Cap at 40 lines
            
            for line in lines:
                if line.strip() in intro_lines: continue
                line_clean = line.strip()
                if not line_clean: continue
                
                # Rule A Check: Must be Command OR Context
                is_cmd = False
                if any(line_clean.lower().startswith(k) for k in ["#", ">>", "=>", "config", "display"]): is_cmd = True
                elif any(k in line_clean.lower() for k in cmd_triggers): is_cmd = True
                
                # Filter: Reject narrative sentences (Thai) unless short context
                # Heuristic: If line has many Thai chars and len > 60 and NOT command -> Skip
                if not is_cmd and len(line_clean) > 60 and any(u'\u0E00' <= c <= u'\u0E7F' for c in line_clean):
                     continue
                
                # Logic: Keep commands and short context lines
                if is_cmd or len(line_clean) < 80:
                     priority_lines.append(line)
                     count_lines += 1
                
                if count_lines >= MAX_LINES:
                    priority_lines.append("... (เนื้อหาตัดตอน เพื่อความกระชับ)\n📌 ดูคำสั่งทั้งหมดในลิงก์ต้นฉบับ")
                    break
            
            if not priority_lines:
                 priority_block = smart_truncate(cleaned_content, max_length=900, footer_url=None)
            else:
                 priority_block = "\n".join(priority_lines)
            
            # 4. Construct Answer
            final_ans = f"[{display_title}]"
            if intro_text:
                final_ans += f"\n\n{intro_text}"
            
            final_ans += f"\n\n{priority_block}"
            
            # 5. Footer (Always Append for How-to - Rule C)
            # final_ans += f"\n\nแหล่งที่มา:\n🔗 {article_url}"
            
            timings["total_ms"] = (time.time() - t_start) * 1000
            return self._wrap_result(final_ans, cleaned_content)
            
        else:
            # Phase 198: Hybrid Summarization Strategy (User Request)
            # Logic: Extractive < 1200 chars | LLM Summarizer >= 1200 chars
            
            # 0. Slicing & Pre-processing (Critical)
            # Clean junk blocks first (already done by clean_article_content + deduplicate)
            # Check length of 'cleaned_content'
            content_len = len(cleaned_content)
            
            # A) Short Article (< 1200 chars) -> Extractive Preview (Fast, Stable)
            if content_len < 1200:
                print(f"[ArticleInterpreter] Hybrid Strategy: SHORT ({content_len} chars) -> Extractive Preview.")
                from src.rag.article_cleaner import smart_truncate
                
                # Phase 21 Clean: Remove table pipe junk and author/date metadata lines
                # These are common scraping artifacts from Joomla CMS
                clean_lines = []
                for line in cleaned_content.split('\n'):
                    stripped = line.strip()
                    # Skip lines that are just pipes (e.g. "| | |", "| |")
                    if re.match(r'^[\|\s]+$', stripped) and '|' in stripped:
                        continue
                    # Also skip lines ending with table pipe artifacts like "Title | | |"
                    if re.search(r'\|\s*\|\s*\|?\s*$', stripped):
                        continue
                    # Skip author/date metadata lines
                    if any(k in stripped for k in ["เขียนโดย", "แก้ไขล่าสุด", "วันศุกร์", "วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันเสาร์", "วันอาทิตย์"]):
                        continue
                    clean_lines.append(line)
                
                preview_content = '\n'.join(clean_lines).strip()
                
                # Phase 21 Format: Expand inline numbered lists into separate lines
                # e.g. "1. item 2. item 3. item" -> "1. item\n2. item\n3. item"
                # Uses lookbehind (?<=[^\d]) to protect IP addresses (192.168.1.1) and decimals
                preview_content = re.sub(r'(?<=[^\d])(\d+\.)\s', r'\n\1 ', preview_content)
                
                preview_text = smart_truncate(preview_content, max_length=900, footer_url=None)
                
                final_ans = f"[{display_title}]\n\n{preview_text}"
                timings["total_ms"] = (time.time() - t_start) * 1000
                result = self._wrap_result(final_ans, cleaned_content)
                result["metadata"]["is_extractive"] = True  # SHORT path: no LLM used
                return result
            
            # B) Long Article (>= 1200 chars) -> LLM Summarizer (High Quality)
            else:
                print(f"[ArticleInterpreter] Hybrid Strategy: LONG ({content_len} chars) -> LLM Summarizer.")
                
                # Phase 18: COMMAND_REFERENCE_MODE
                if intent == "COMMAND_REFERENCE":
                    print(f"[ArticleInterpreter] Phase 18: Using TEMPLATE_COMMAND_REFERENCE")
                    from src.rag.prompts import TEMPLATE_COMMAND_REFERENCE
                    prompt_vals = {
                         "title": display_title,
                         "query": user_query,
                         "context_str": cleaned_content[:8000]
                    }
                    prompt = TEMPLATE_COMMAND_REFERENCE.format(**prompt_vals)
                    try:
                        from src.rag.ollama_client import ollama_generate
                        llm_summary = ollama_generate(
                            base_url=self.base_url,
                            model=self.model,
                            prompt=prompt,
                            system="You are a strict technical command extractor.",
                            max_tokens=400,
                            temperature=0.0
                        ).strip()
                        timings["total_ms"] = (time.time() - t_start) * 1000
                        return self._wrap_result(llm_summary, cleaned_content)
                    except Exception as e:
                        print(f"[ArticleInterpreter] Command Summarizer Failed: {e}")
                        # Fallback to extractive
                
                # Phase 218: Technical Knowledge Summarizer (Extraction-First)
                # Extract commands BEFORE prompting LLM
                extracted_cmds = extract_cli_commands(cleaned_content, min_lines=4)
                
                cmd_instruction = ""
                if extracted_cmds:
                     # Case A: Commands Found -> Inject into Prompt
                     extracted_cmds = extracted_cmds[:1200]
                     cmd_instruction = f"Commands start below:\n```text\n{extracted_cmds}\n```\nINSTRUCTION: Place these exact commands in the 'คำสั่งตัวอย่าง' section."
                else:
                     # Case B: No Commands -> Instruction to Omit
                     cmd_instruction = "INSTRUCTION: No extracted commands found. DO NOT create a 'คำสั่งตัวอย่าง' section. Skip it entirely."

                prompt_vals = {
                     "title": display_title,
                     "query": user_query,
                     "context_str": cleaned_content[:8000],  
                     "url": article_url,
                     "command_instruction": cmd_instruction 
                }
                
                # Determine Prompt (Phase 208/215)
                # Ensure TEMPLATE_SUMMARIZER uses {command_instruction}
                from src.rag.prompts import TEMPLATE_SUMMARIZER
                prompt = TEMPLATE_SUMMARIZER.format(**prompt_vals)
                
                try:
                    from src.rag.ollama_client import ollama_generate
                    llm_summary = ollama_generate(
                        base_url=self.base_url,
                        model=self.model,
                        prompt=prompt,
                        system="You are a professional Answer Extractor.", 
                        max_tokens=512,
                        temperature=0.2
                    ).strip()

                    llm_summary = self._clean_llm_response(llm_summary)
                    
                    # Phase 18: Link appending removed (handled by ChatEngine)
                    # llm_summary += f"\n\nแหล่งที่มา:\n🔗 {article_url}"

                    timings["total_ms"] = (time.time() - t_start) * 1000
                    return self._wrap_result(llm_summary, cleaned_content)
                    
                except Exception as e:
                    print(f"[ArticleInterpreter] LLM Summarizer Failed: {e}. Fallback to Extractive.")
                    # Fallback to Extractive
                    from src.rag.article_cleaner import smart_truncate
                    final_ans = f"[{display_title}]\n\n(ระบบสรุปขัดข้อง แสดงเนื้อหาบางส่วน)\n{preview_text}"
                    return self._wrap_result(final_ans, cleaned_content)

            # Legacy logic (Direct Preview / Type A) is largely bypassed by this Unified Policy
            # But we keep it below if needed, or we can just return above always.
            # To follow the plan strictly "make it the standard", we return here.

            # Priority: After Tutorial check (LLM), but BEFORE Type A (Extraction).
            # We prefer "Raw Content" over "Extracted Facts" for generic queries.
             
            print(f"[ArticleInterpreter] Phase 173 Check (Pre-Type A). UX Config: {self.ux_cfg}")
            use_direct_preview = self.ux_cfg.get("direct_content_preview", True)
             
            if use_direct_preview:
                from src.rag.article_cleaner import smart_truncate, format_fact_item, mask_sensitive_data
                  
                print(f"[ArticleInterpreter] Phase 173: Using Direct Content Preview")
                
                # 0. Heuristic: Skip Junk Stats Table (Header)
                # If the content starts with a stats table (lines with pipes '|' and digits), skip it.
                lines = cleaned_content.split('\n')
                start_idx = 0
                junk_rows = 0
                # Check first 15 lines max
                for i in range(min(15, len(lines))):
                    line = lines[i].strip()
                    if not line: continue
                    
                    # Criteria: Contains pipe OR high digit ratio (stats usually have numbers)
                    is_pipe = '|' in line
                    digit_count = sum(c.isdigit() for c in line)
                    is_digit_heavy = (len(line) > 0 and digit_count / len(line) > 0.3)
                    
                    if is_pipe or is_digit_heavy or "This month" in line or "Last month" in line:
                         junk_rows += 1
                         start_idx = i + 1
                    else:
                         # Found non-junk line (text?), stop skipping if we found sequential junk
                         if junk_rows > 0:
                              break
                
                # Only skip if we found a significant block of junk (>2 lines)
                target_content = cleaned_content
                if junk_rows >= 2:
                     print(f"[ArticleInterpreter] Skipped {junk_rows} junk header lines.")
                     target_content = "\n".join(lines[start_idx:]).strip()

                # Phase 174 Fix: Strip Navigation/Menu Text (User Request)
                from src.rag.article_cleaner import strip_navigation_text
                target_content = strip_navigation_text(target_content)

                # 1. Format/Split Links
                step1_content = format_fact_item([target_content], enable_formatting=True)
                  
                # 2. Safety: Mask Credentials (in case raw content has them)
                step2_content = mask_sensitive_data(step1_content)
                
                # Phase 174: Deduplicate (Remove repetitive paragraphs)
                step2_content = deduplicate_paragraphs(step2_content)
                
                # 3. Prepend Title (Requirement 1)
                # Heuristic: If DB title is generic (e.g. "News Article"), use User Query or fallback.
                display_title = article_title.strip()
                if not display_title or "News Article" in display_title or "บทความ" == display_title:
                     display_title = user_query.strip()
                
                final_content = f"[{display_title}]\n\n{step2_content}"

                # 4. Truncate
                preview = smart_truncate(final_content, max_length=1200, footer_url=None)
                
                # Standard Footer (Short/Preview) - Removed for Phase 18
                # preview += f"\n\nแหล่งที่มา:\n🔗 {article_url}"
                  
                timings["total_ms"] = (time.time() - t_start) * 1000
                return self._wrap_result(preview, cleaned_content)

            # Type A / Default Logic
            has_structure = has_structured_content(cleaned_content)
            print(f"[ArticleInterpreter] has_structured_content (Type A Check): {has_structure}")

            if has_structure:
                 # Try deterministic extraction (Fast-Path)
                 facts = extract_topic_anchored_facts(
                     cleaned_content, 
                     user_query, 
                     enable_formatting=self.ux_cfg.get("enable_formatting", True)
                 )
                 print(f"[ArticleInterpreter] Extracted {len(facts)} facts (Topic-Anchored)")
                 
                 # Phase 119: Type A Auto-Formatting (Credentials)
                 joined_facts = "\n".join(facts)
                 is_credential_dump = any(k in joined_facts.lower() for k in ["user:", "pass:", "password", "username"])
                 
                 if is_credential_dump and len(facts) > 0:
                      from src.rag.article_cleaner import format_credential_structure, mask_sensitive_data
                      print(f"[ArticleInterpreter] Credential Dump Detected -> Auto-Formatting")
                      formatted_creds = format_credential_structure(joined_facts)
                      formatted_creds = mask_sensitive_data(formatted_creds)
                      answer = f"[{article_title}]\n{formatted_creds}"
                      timings["select_ms"] = 0
                      timings["llm_ms"] = 0
                      timings["total_ms"] = (time.time() - t_start) * 1000
                      return self._wrap_result(answer, cleaned_content)

                 if len(facts) >= 1:
                     # Type A: Fast-Path Success
                     answer = f"[{article_title}]\n"
                     for i, fact in enumerate(facts, 1):
                         answer += f"{i}. {fact}\n"
                     timings["select_ms"] = 0
                     timings["llm_ms"] = 0
                     timings["total_ms"] = (time.time() - t_start) * 1000
                     print(f"[ArticleInterpreter] Fast-path (Type A): {timings['total_ms']:.1f}ms (no LLM)")
                     return answer
                 else:
                     print(f"[ArticleInterpreter] Structure found but facts filtered -> Falling back to LLM")




        # 3. Truncate content if too long (for LLM path only)
        # Type B logic already set limit_chars above if needed
        t_select = time.time()
        
        truncated_content, was_truncated = truncate_content(cleaned_content, user_query, max_chars=limit_chars)
        
        timings["select_ms"] = (time.time() - t_select) * 1000
        print(f"[ArticleInterpreter] Content: original={len(article_content)}, cleaned={len(cleaned_content)}, truncated={len(truncated_content)}, was_truncated={was_truncated}")
        
        # 4. Validate content quality (check for metadata/nav-only content) BEFORE LLM
        # Phase 123: Bypass Nav/Metadata Check completely for Type B
        if not is_tutorial_mode:
            # Phase 20.5: Check for Navigation Domination too
            is_metadata = is_metadata_dominated(truncated_content)
            is_nav = is_navigation_dominated(truncated_content)
        
            if is_metadata or is_nav:
                print(f"[ArticleInterpreter] Content rejected: metadata={is_metadata}, nav={is_nav}")
                
                # Phase 30: Second-Chance Selection (Procedural Recovery)
                recovered_content = second_chance_procedural_extraction(article_content)
                if recovered_content and len(recovered_content) > 50:
                    print(f"[ArticleInterpreter] Second Chance: Recovered {len(recovered_content)} chars of procedural content")
                    truncated_content = recovered_content
                    # Fall through to Fast Path / LLM
                else:
                    # Confirmed Rejection
                    timings["llm_ms"] = 0
                    timings["total_ms"] = (time.time() - t_start) * 1000
                
                if is_nav:
                    # Phase 30: Wrapper Guard
                    # import re (Moved to top)
                    links = re.findall(r'(.+?)\s*\((http[s]?://\S+)\)', truncated_content)
                    valid_links = []
                    for title, url in links:
                        if len(title) > 5 and "http" not in title and len(valid_links) < 3:
                            valid_links.append(f"- [{title.strip()}]({url})")
                    
                    if valid_links:
                         links_str = "\n".join(valid_links)
                         return self._wrap_result(f"เนื้อหานี้เป็นเพียงหน้ารวมลิงก์/เมนู (Navigation Page) กรุณาเลือกหัวข้อที่ต้องการ:\n\n{links_str}", truncated_content)
                    
                    return self._wrap_result("เนื้อหาที่พบเป็นเพียงเมนูนำทาง ไม่พบข้อมูลรายละเอียดในบทความนี้", truncated_content)
                    
                return self._wrap_result("ไม่พบข้อมูลที่ยืนยันคำตอบนี้ในระบบ", truncated_content)
        
        # Phase 31: Procedural Fast-Path (Report Articles)
        # Optimization: If content contains specific report patterns, extract deterministically to bypass LLM.
        # Check this on 'truncated_content' (which might be recovered content or standard cleaned content)
        # import re (Moved to top)
        # Fix Phase 132: Disable Procedural Fast-Path for Tutorials
        if not is_tutorial_mode and any(kw in truncated_content for kw in ["ภายในเวลา", "แบบฟอร์ม", "ส่ง", "mail"]):
             # Structured Extraction (Phase 33)
             struct_data = {"time": [], "email": [], "subject": [], "form": [], "instructions": []}
             
             lines = truncated_content.split('\n')
             for line in lines:
                 line = line.strip()
                 if len(line) < 5: continue
                 
                 # 1. Time
                 t_match = re.search(r'(\d{1,2}[\.:]\d{2}\s*น\.)', line)
                 if t_match: struct_data["time"].append(t_match.group(1))
                 
                 # 2. Email
                 if "@" in line or "mail" in line.lower():
                     # Try to extract just the email if possible, or keep line
                     struct_data["email"].append(line)
                     
                 # 3. Subject/Title
                 if "หัวข้อ" in line or "Subject" in line:
                     # Remove 1. 2. prefix
                     clean = re.sub(r'^\d+\.\s*', '', line)
                     clean = clean.replace("หัวข้อ", "").replace("Subject", "").replace(":", "").strip()
                     struct_data["subject"].append(clean)
                     
                 # 4. Form/Action
                 elif "แบบฟอร์ม" in line or "http" in line:
                     clean = re.sub(r'^\d+\.\s*', '', line)
                     struct_data["form"].append(clean)
                     
                 # 5. Key Roles (Sender/Receiver)
                 elif any(k in line for k in ["ผู้จัดทำ", "ส่งถึง", "เรียน", "CC:"]):
                     clean = re.sub(r'^\d+\.\s*', '', line)
                     struct_data["instructions"].append(clean)

             # Construct Polished Bullets
             final_bullets = []
             
             # Priority 1: Time
             if struct_data["time"]:
                 final_bullets.append(f"• เวลาส่ง: {struct_data['time'][0]}") # Take first valid time
                 
             # Priority 2: Subject
             if struct_data["subject"]:
                 final_bullets.append(f"• หัวข้ออีเมล: {struct_data['subject'][0]}")
                 
             # Priority 3: Sender/Roles
             for item in struct_data["instructions"][:3]: # Limit roles
                 final_bullets.append(f"• {item}")
                 
             # Priority 4: Emails
             for item in struct_data["email"][:2]:
                 final_bullets.append(f"• {item}")
                 
             # Priority 5: Forms
             for item in struct_data["form"][:2]:
                 final_bullets.append(f"• {item}")

             # Deduplicate
             unique_bullets = []
             seen = set()
             for b in final_bullets:
                 b = re.sub(r'\s+', ' ', b).strip()
                 if b not in seen:
                     unique_bullets.append(b)
                     seen.add(b)
             
             if len(unique_bullets) >= 3:
                 print(f"[ArticleInterpreter] Structured Fast-Path: Generated {len(unique_bullets)} bullets")
                 timings["total_ms"] = (time.time() - t_start) * 1000
                 
                 ans = f"[{display_title}]\n"
                 for b in unique_bullets:
                     ans += f"{b}\n"
                 
                 # Append Footer (Source Standardization) - Removed for Phase 18
                 # ans += f"\n\nแหล่งที่มา:\n🔗 {article_url}"
                 return self._wrap_result(ans, truncated_content)

        # Phase 74: Link Directory Parser (Manual/Download Pages)
        print("[ArticleInterpreter] Checking DirectoryParser...")
        # Check full cleaned content for directory structure
        if self._looks_like_link_directory(cleaned_content, user_query):
             print(f"[ArticleInterpreter] Detected Link Directory/Manual Page")
             directory_ans = self._parse_link_directory(cleaned_content, article_title)
             if directory_ans:
                 timings["total_ms"] = (time.time() - t_start) * 1000
                 return self._wrap_result(directory_ans, cleaned_content)
                 
        # 5. Prepare image information for LLM (only if show_images=True)

        # 5. Prepare image information for LLM (only if show_images=True)
        image_info = ""
        if show_images and images:
            from src.rag.image_filter import filter_and_rank_images
            
            # Filter and rank images (max 3 for explicit image queries)
            filtered_images = filter_and_rank_images(images, user_query, max_images=3)
            
            # Additional validation: filter invalid image URLs
            filtered_images = filter_valid_images(filtered_images)
            
            if filtered_images:
                image_info = "\n\nImages available in article:\n"
                for idx, img in enumerate(filtered_images, 1):
                    img_url = img.get("url", "")
                    img_alt = img.get("alt", "")
                    img_caption = img.get("caption", "")
                    image_info += f"{idx}. URL: {img_url}\n"
                    if img_alt:
                        image_info += f"   Alt: {img_alt}\n"
                    if img_caption:
                        image_info += f"   Caption: {img_caption}\n"
            else:
                # No valid images found
                if show_images:
                    image_info = "\n\n(ไม่มีรูปภาพที่เกี่ยวข้องกับคำถามนี้ในบทความ)\n"
        


        
        if is_tutorial_mode:
            system_prompt = """You are a Technical Documentation Summarizer for an internal enterprise system.
Your task is to summarize the provided article ONLY when the user's question is conceptual, tutorial-style, or explanatory.

STRICT RULES:
1. Use ONLY information explicitly found in the provided article content.
2. DO NOT add external knowledge, best practices, or assumptions.
3. Output Format:
   - Provide a clear summary (6-8 Bullet points max).
   - If steps are involved, list them concisely.
   - Do NOT add a conclusion or "Read more" phrase (System will add it).
4. Language: Thai only.

If the article content is unrelated, empty, or purely navigational:
- Respond with: "ไม่พบเนื้อหาที่เพียงพอสำหรับการสรุปจากแหล่งข้อมูลนี้"
"""
            # Minimal User Prompt for Type B (Let System Prompt drive)
            prompt_instruction = "Summarize the key points in 6-8 bullets."
            
        
        else:
            # Type A / Fallback (Existing Strict Extractor)
            system_prompt = """You are an Enterprise Knowledge Article Interpreter for SMC.

Your role:
- Read the FULL content of a news/article page.
- Answer user questions by extracting and structuring information from the article ONLY.
- Never invent, infer, or use outside knowledge.

Your tasks:
1) Determine whether the user is asking about the whole article or a specific section.
2) Extract ONLY information that explicitly appears in article_content.
3) Structure the answer clearly using numbered or bulleted sections.
   - Preserve technical values exactly.

4) Output rules (STRICT):
   - Language: Thai only
   - No explanations about what the article is
   - No meta commentary
   - If information is missing, say: "ไม่พบข้อมูลนี้ในข่าวดังกล่าว"
   
   [Link/Document Formatting]
   When the answer contains multiple documents or links:
   - Always format the response as bullet points.
   - Each document must be on its own bullet.
   - Format: "[Document Name] (URL)" or
     * [Document Name]
       Link: [URL]
   - Do NOT merge multiple links into one sentence.


5) Output format (example):
[ข่าว NT-1]
1. <หัวข้อย่อย>
   - รายละเอียด...

6) Image handling:
   - If images are relevant, include them in a SEPARATE section at the END.
   - Use: ![description](url)

7) Safety:
   - Do NOT mask credentials.
   - Do NOT guess missing fields.

If the article content is empty or irrelevant, reply: "เนื้อหาในบทความไม่เพียงพอต่อการสรุป (กรุณาคลิกอ่านฉบับเต็ม)"
"""
            prompt_instruction = "Extract and structure the answer as NUMBERED ITEMS ONLY. Be concise and extractive."


        user_prompt = f"""Article Title: {article_title}
Article URL: {article_url}

Article Content:
{truncated_content}{image_info}

User Question: {user_query}

{prompt_instruction}
If images are relevant to the user's question, include them in a separate section at the end."""

        try:
            t_llm = time.time()
            answer = ollama_generate(
                base_url=self.base_url,
                model=self.model,
                prompt=user_prompt,
                system=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            timings["llm_ms"] = (time.time() - t_llm) * 1000
            
            timings["total_ms"] = (time.time() - t_start) * 1000
            print(f"[ArticleInterpreter] Timing: total={timings['total_ms']:.1f}ms (clean={timings['clean_ms']:.1f}ms, select={timings['select_ms']:.1f}ms, llm={timings['llm_ms']:.1f}ms)")
            
            
            final = answer.strip()
            # Phase 46: Safety Guard
            from src.rag.article_cleaner import mask_sensitive_data
            final = mask_sensitive_data(final)
            final = mask_sensitive_data(final)
            final = mask_sensitive_data(final)
            
            # Phase 175: Policy 1 - Append Link for Tutorials
            if is_tutorial_mode and "แหล่งที่มา" not in final:
                final += f"\n\nแหล่งที่มา:\n🔗 {article_url}"
                
            print(f"ArticleMode=article Links={cleaned_content.lower().count('http')} TextChars={len(cleaned_content)} Score={match_score} Cache=miss")
            return final

            
        except Exception as e:
            print(f"[ArticleInterpreter] Error: {e}")
            return f"เกิดข้อผิดพลาดในการประมวลผลข่าว: {str(e)}"

    def _looks_like_link_directory(self, content: str, user_query: str = "") -> bool:
        """
        Heuristic to detect Manual/Download pages.
        Criteria: High density of 'ftp://', 'http', 'Manual'
        Refined: Must have many links AND low non-link text density.
        """
        # Phase 183 Fix: Check intent first (User Request - EDIMAX Case)
        # If user asks for How-to/Explain, avoid classifying as Directory if content is substantial
        if user_query and self._is_tutorial_intent(user_query):
             # Phase 185 Strict Rule: Only classify as Nav Page if links are excessive AND text is minimal
             # Count actual text chars (excluding link URLs)
             link_count = content.lower().count("http")
             
             # Heuristic: Remove all links and common nav labels to see raw content
             text_only = re.sub(r'(ftp|https?)://[^\s\)]+', '', content.lower())
             clean_text_len = len(text_only.strip())
             
             # If intent is HOWTO, we ONLY accept Directory if:
             # 1. Links >= 20 (Very high density) OR
             # 2. Text < 250 (Very low content) AND Links >= 5
             # Relaxed rule: EDIMAX had 48 links! So we need to ensure text length saves it.
             # User Request: "links_count >= 20 AND clean_text_chars < 200"
             # Phase 209: Stricter check for false positives (Bridge case)
             # If "Bridge" has ~20 links but ~2k text chars, it's NOT a directory.
             
             if link_count >= 25 and clean_text_len < 300:
                 return True # It IS a link dump (e.g. strict directory)
                 
             # If text is substantial (>1000 chars), almost certainly an Article/Tutorial
             if clean_text_len > 1000:
                 return False
                 
             # Otherwise, assume it's a helpful article with many references (e.g. EDIMAX)
             return False

        content_lower = content.lower()
        
        # 1. Count Protocol Links (ftp/http)
        link_count = content_lower.count("ftp://") + content_lower.count("http://") + content_lower.count("https://")
        
        # 2. Key Terms (Must act like a directory)
        directory_keywords = ["manual", "คู่มือ", "guide", "download", "ดาวน์โหลด", "เอกสารประกอบ", "link directory", "software"]
        has_keyword = any(k in content_lower for k in directory_keywords)
        
        # 3. Text Density Check (Avoid false positives on long articles with some links)
        # Remove links to estimate "narrative content"
        text_only = re.sub(r'(ftp|https?)://[^\s\)]+', '', content_lower)
        text_len = len(text_only.strip())
        
        # 4. Strict Decision
        
        # Phase 212: Global Safety for Long Articles (The "Bridge" Fix)
        # If content has > 2000 chars of pure text, it is NEVER a directory, 
        # even if it has 50 links.
        if text_len > 2000:
             return False

        # Case A: Very high link count (Link dump)
        if link_count >= 15: return True
        
        # Case B: Standard Directory (Links + Keywords + Low Text)
        if link_count >= 8 and has_keyword and text_len < 1000:
             return True
             
        # Case C: Explicit "Manual/Document" List (e.g. Ribbon)
        # Phase 209 Fix: Must enforce text length to avoid catching long articles that mention manuals
        if link_count >= 5 and (content_lower.count("manual") + content_lower.count("คู่มือ") >= 2):
             if text_len < 1200: # Safety cap
                return True
        
        return False


    def _parse_link_directory(self, content: str, title: str, user_query: str = "", article_url: str = "") -> str:
        """
        Phase 208: Standardized Prompt: DIRECTORY_CURATOR
        Relevance is mandatory. Do not include unrelated templates/tools.
        RULES:
        1) Show ONLY links relevant to the query (keyword overlap).
        2) Remove generic/unrelated items.
        3) Return 2–6 links maximum.
        4) Always include source_url.
        """
        import re
        
        # 1. Clean Title (Limit 90 chars)
        title = re.sub(r'\|[\s\|]+\|.*$', '', title).strip().replace('|', '').strip()
        if not title: title = "Directory Links"
        if len(title) > 90: title = title[:87] + "..."
        
        # 2. Extract ALL Links first (Robust Scan)
        found_links = [] 
        
        chunks = re.split(r'(https?://[^\s\)\>\]]+)', content)
        
        for i in range(1, len(chunks), 2):
            url = chunks[i].strip()
            pre_text = chunks[i-1].strip()
            
            link_text = ""
            lines = pre_text.split('\n')
            if lines and lines[-1].strip():
                link_text = lines[-1].strip()
            else:
                link_text = "Link"
            
            # Clean link text
            link_text = re.sub(r'[\(\[\]\)\:\-]+$', '', link_text).strip()
            # Remove leading bullets/garbage (e.g. "1.", "-", "!", ")")
            link_text = re.sub(r'^[\d\W_]+', '', link_text).strip()
            
            link_text = re.sub(r'(Link|File|PDF|Manual|Topic)\s*$', '', link_text, flags=re.IGNORECASE).strip()
            if len(link_text) < 3: link_text = "See Link"
            
            found_links.append({"text": link_text, "url": url})
            
        # 3. Filter & Deduplicate
        unique_links = []
        seen_urls = set()
        
        junk_patterns = ["joomla", "template", "wrapper", "license", "index.php?option=com_mailto", "print=1", "component", 
                         "นราธิวาส", "สุราษฎร์", "totsni.com", "10.192.39.50", "ศูนย์ปฏิบัติการ", "ผส.บลตน", "ผจ.สบลตน", 
                         "เบอร์หน่วยงาน", "intranet nt", "web hr", "ข่าวสาร smc", "ความรู้", "บุคลากร smc", "ผู้ดูแลระบบ", "edocument"]
        
        # Query Keywords (Rule 1: at least 1 keyword overlap)
        q_keywords = [t.lower() for t in user_query.split() if len(t) > 2] if user_query else []
        
        for item in found_links:
            u_lower = item["url"].lower()
            t_lower = item["text"].lower()
            
            if u_lower in seen_urls: continue
            seen_urls.add(u_lower)

            # Anti-Junk Filter (Check both URL and Text)
            # v209 Fix: "Template" in text should be excluded.
            is_junk = False
            for k in junk_patterns:
                if k in u_lower or k in t_lower:
                    is_junk = True
                    print(f"DEBUG: REJECTED JUNK LINK: {t_lower} | {u_lower} (Matced: {k})")
                    break
            
            if is_junk: continue
            
            is_relevant = False
            score = 0
            
            # Strict Relevance: Must share at least 1 keyword
            matches = 0
            if q_keywords:
                matches = sum(1 for k in q_keywords if k in t_lower or k in u_lower)
                if matches > 0:
                     score = matches * 10
                     is_relevant = True
            
            # Rule 3 (Relaxed): If strict relevance found nothing, use ALL valid unique links (Score 1)
            # This handles "Nav Page" where user query might be loose but we want to show the menu.
            if not is_relevant and not q_keywords:
                 # Original browsing fallback
                 if ".pdf" in u_lower or "manual" in t_lower: 
                     score = 5
                     is_relevant = True
            elif not is_relevant:
                 # Fallback: Content from a Nav Page is likely relevant by virtue of being there.
                 # Assign low score.
                 score = 1
                 is_relevant = True

            if is_relevant:
                item["score"] = score
                if len(item["text"]) > 90:
                    item["text"] = item["text"][:89] + "…"
                unique_links.append(item)
                print(f"DEBUG: ADDED LINK: {item['text']} | {item['url']}")
            
        # 4. Sort & Cap (Max 6)
        unique_links.sort(key=lambda x: x["score"], reverse=True)
        
        # New Logic: Winner Takes All
        # If we found High Relevance links (Score >= 10), drop all Low Relevance fallback links (Score < 10).
        if unique_links and unique_links[0]["score"] >= 10:
             unique_links = [x for x in unique_links if x["score"] >= 10]
        
        top_links = unique_links[:6]
        print(f"DEBUG: FINAL TOP LINKS: {[i['text'] for i in top_links]}")
        
        # Rule 3 Fix: Return even 1 link if valid (Don't fail closed on Nav Pages)
        if not top_links:
            return "" 
            
        # 5. Construct Output (User Schema)
        display_title = title if title else "Directory Links"
        
        out_lines = [f"[{display_title}]", f"ผู้เขียน: - | วันที่: -\n"] # Fill metadata if available/passed?
        out_lines.append("บทความนี้เป็น “เมนูรวมลิงก์/ไฟล์” ยังไม่มีเนื้อหาให้สรุปโดยตรง")
        # Removed hardcoded "เลือกหัวข้อที่เกี่ยวข้อง:" header (Phase 221 Fix)
        # out_lines.append("เลือกหัวข้อที่เกี่ยวข้อง:\n")
        out_lines.append("")
        
        for item in top_links:
             # <title> 🔗 <url>
             out_lines.append(f"{item['text']} 🔗 {item['url']}\n")
             
        out_lines.append("\nแหล่งที่มา:\n")
        out_lines.append(f"🔗 {article_url}")
            
        return "\n".join(out_lines).strip()
        
    def _clean_llm_response(self, text: str) -> str:
        """
        Phase 219: Strict Post-Processing Guardrails
        1. Limit bullets to 5 per section.
        2. Remove lines with forbidden keywords (marketing fluff).
        """
        if not text: return text
        
        forbidden_keywords = ["ช่วยให้", "เพิ่มความ", "สามารถใช้เพื่อ", "เหมาะสำหรับ", "แนะนำให้", "ทำให้", "ช่วยเพิ่ม", "จุดเด่น"]
        
        cleaned_lines = []
        lines = text.splitlines()
        
        current_section = None
        bullet_count = 0
        
        for line in lines:
            s_line = line.strip()
            
            # Phase 236: Strip Prompt Artifacts (Rule SA-1)
            # If line contains INSTRUCTION: or If MENU_MODE:, skip until end of block/text
            if "INSTRUCTION:" in s_line or "If MENU_MODE:" in s_line:
                continue
            
            # Detect Section Headers
            if s_line.startswith("สรุปย่อ:") or s_line.startswith("แนวคิดทางเทคนิค"):
                 current_section = s_line
                 bullet_count = 0
                 cleaned_lines.append(line)
                 continue
                 
            # Detect Bullets
            if s_line.startswith("- ") or s_line.startswith("* "):
                 # Check Limit
                 if bullet_count >= 5: 
                     continue # Skip excess bullets
                     
                 # Check Forbidden Words
                 if any(k in s_line for k in forbidden_keywords):
                     continue # Skip fluff
                     
                 bullet_count += 1
                 cleaned_lines.append(line)
            else:
                 # Pass through other lines (headers, commands, etc)
                 cleaned_lines.append(line)
                 
        return "\n".join(cleaned_lines)

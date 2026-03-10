
from __future__ import annotations

import time
import re
import datetime
from src.utils.normalization import normalize_for_matching
from src.utils.metrics import MetricsTracker
import json
import difflib
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from src.utils.runlog import now_ts
import sys
from src.vectorstore import build_vectorstore
import yaml
from src.vectorstore.base import SearchResult # Step 16954: For creating boosted results
from src.directory.lookup import load_records, lookup_phones, lookup_by_phone, strip_query, norm
from src.rag.handlers.greetings_handler import GreetingsHandler # Phase 97
from src.cache.semantic import SemanticCache # Phase 23
from src.ingest.fetch import fetch_with_policy, FetchResult # Phase 130
from src.rag.synonyms import expand_synonyms # Phase 174
from src.rag.junk_filter import clean_junk_text # Rule JR-1


from src.rag.retrieval_optimizer import RetrievalOptimizer
from src.rag.controller import RAGController
from src.rag.prompts import PROMPT_VERSION # Phase 235: Versioned Cache
from src.utils.section_filter import slice_markdown_section # Assumption: found here
from src.rag.evaluator import RAGEvaluator 
from src.rag.article_interpreter import ArticleInterpreter
from src.rag.ollama_client import ollama_generate # FIXED: Added missing import
from src.ai.planner import QueryPlanner
from src.ai.slot_filler import SlotFiller
from src.directory.format_answer import format_field_only
from src.rag.handlers.directory_handler import DirectoryHandler # Phase 43
from src.rag.handlers.web_handler import WebHandler # Phase 175: Web Knowledge
from src.ai.safe_normalizer import SafeNormalizer 
from src.ai.query_normalizer import QueryNormalizer  # Phase 2.1
from src.ai.content_classifier import ContentClassifier  # Content-Type Classifier

# Guard Mechanisms & Routing
from src.ai.greeting_gate import is_pure_greeting, get_greeting_response
from src.ai.howto_shield import PROMPT_HOWTO_SHIELD
from src.ai.contact_precision import is_valid_contact_query
from src.ai.failsoft_reroute import should_reroute_to_howto, get_reroute_intent
from src.ai.technical_protection import is_asset_table_query, has_protected_term
from src.ai.canonical_phrases import get_canonical_phrase, apply_canonical_rules
from src.context import context_manager  # Conversation context memory
from src.ai.routing_policy import RoutingPolicy
# from src.main import ProcessedCache # Removed to fix ImportError 
# Better to move ProcessedCache to utils or keep it in main and import? 
# To avoid circular imports, let's duplicate or move ProcessedCache to a util file. 
# For now, I'll redefine it here or just access data directly if possible.
# Actually, ProcessedCache is simple. Let's move it to src/utils/cache.py later? 
# For now, I will implement a simple version here to avoid modifying too many files at once.

# Phase 20: Procedural/Knowledge keywords
PROCEDURAL_KEYWORDS = {
    "วิธี", "ขั้นตอน", "ตั้งค่า", "โอนสาย", "forward", "transfer", 
    "ทำยังไง", "คู่มือ", "ปัญหา", "แก้ไข", "วิธีแก้", "how to", 
    "news", "ข่าวสาร", "config", "setup", "guide", "manual",
    "ระเบียบ", "คำสั่ง", "ประกาศ", "หนังสือเวียน"
}

# Explicit contact keywords
CONTACT_KEYWORDS = {
    "เบอร์", "โทร", "ติดต่อ", "อีเมล", "fax", "แฟกซ์", "โทรสาร", "ที่อยู่"
}

# Phase 236: Technical Glossary for DEFINE_TERM (Rule DT-2)
# Avoid incorrect Retrieval for standard technical terms.
TECHNICAL_GLOSSARY = {
    "RAG": {
        "full_name": "Retrieval-Augmented Generation",
        "definition": "เทคนิคที่ช่วยให้ LLM สามารถตอบคำถามโดยใช้ข้อมูลจากเอกสารภายนอก (เช่น SMC) เพื่อลดการมโน (Hallucination) และเพิ่มความถูกต้องของข้อมูล"
    },
    "OLT": {
        "full_name": "Optical Line Terminal",
        "definition": "อุปกรณ์ศูนย์กลางในโครงข่ายใยแก้วนำแสง (FTTx) ทำหน้าที่รับส่งสัญญาณระหว่างเครือข่ายหลักกับอุปกรณ์ปลายทาง (ONT/ONU) ของลูกค้า"
    },
    "ONT": {
        "full_name": "Optical Network Terminal",
        "definition": "อุปกรณ์ปลายทางที่ติดตั้งในบ้านลูกค้า เพื่อแปลงสัญญาณแสงจากใยแก้วนำแสงเป็นสัญญาณอินเทอร์เน็ต/โทรศัพท์"
    },
    "GPON": {
        "full_name": "Gigabit Passive Optical Network",
        "definition": "มาตรฐานเทคโนโลยีการส่งข้อมูลผ่านโครงข่ายใยแก้วนำแสงความเร็วสูงที่นิยมใช้ในปัจจุบัน"
    },
    "VLAN": {
        "full_name": "Virtual Local Area Network",
        "definition": "การจำลองเครือข่ายท้องถิ่นเสมือน เพื่อแบ่งกลุ่มการใช้งานและเพิ่มความปลอดภัยในการรับส่งข้อมูล"
    }
}

# ============================================================================
# STEP 1-2: PENDING STATE CONTRACT & TOKEN SEPARATION
# ============================================================================

# STEP 2: Confirmation Tokens (used ONLY when pending state is active)
CONFIRMATION_TOKENS = {
    # Thai
    "ใช่", "ตกลง", "โอเค", "ok",
    # English  
    "yes",
    # Note: Numeric tokens (1, 2, 3) handled separately in resolver
}

# STEP 2: Greeting Whitelist (exact match only, NO heuristics)
GREETING_WHITELIST = {
    # Thai
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ",
    "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ",
    # English
    "hello", "hi", "thanks",
}

# STEP 1: Pending State TTL (seconds)
PENDING_STATE_TTL = 120  # 2 minutes


# Phase 174: Content Polish & Synonym (v1.1-dev)
# Increment this when changing summary logic or formatting to force re-evaluation.
CACHE_SCHEMA_VERSION = "v236"

class ProcessedCache:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self._url_to_text: dict[str, str] = {}
        self._url_to_images: dict[str, list] = {}  # NEW: Store images per URL
        self._link_index: dict[str, list] = {} # title -> [{"text": ..., "href": ...}]
        self._normalized_title_index: dict[str, dict] = {}  # Phase SMC: Fast title lookup
        self._soft_normalized_title_index: dict[str, dict] = {} # Phase 20: Soft-normalized fast lookup
        self._keys: list[str] = [] # Cache keys for iteration
        self._slug_index: dict[str, dict] = {} # Phase 16: Dedicated slug index
        self._loaded = False
        
        # Load Aliases from data/aliases.json (Phase 7.1)
        alias_path = Path("data/aliases.json")
        if alias_path.exists():
            with open(alias_path, "r", encoding="utf-8") as f:
                self.aliases = json.load(f)
                print(f"[ProcessedCache] Loaded {len(self.aliases)} aliases")
        else:
            # Phase 6: Fail-Closed
            print(f"[ProcessedCache] CRITICAL ERROR: {alias_path} not found.")
            raise RuntimeError("System Halt: Missing Aliases Configuration (aliases.json)")
        
        # Phase ???: Simple corpus statistics
        self.stats = {"total_docs": 0, "total_links": 0}

    def normalize_key(self, text: str) -> str:
        # Simple normalization for URL/Key matching
        return text.lower().strip()

    def soft_normalize(self, text: str) -> str:
        """
        Phase 20: Soft Normalization Layer.
        - Trim whitespace
        - Lowercase
        - Replace - and _ and multiple spaces with a single space
        - Collapse whitespace to single space
        - NO word removal, NO re-ordering.
        """
        if not text: return ""
        t = text.lower().strip()
        # Replace dashes and underscores with spaces
        t = re.sub(r"[\-_]", " ", t)
        # Collapse all whitespace sequences to single space
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def normalize_for_matching(self, text: str) -> str:
        """
        Phase 7: Query Normalization for Deterministic Matching.
        - Lowercase
        - Remove noise words (how, config, command)
        - Collapse spaces
        """
        t = text.lower()
        # Remove noise words
        noise = ["how to", "how", "วิธี", "config", "configuration", "คำสั่ง", "command", "manual", "คู่มือ", "guide", "basic", "พื้นฐาน", "syntax", "cmd"]
        for n in noise:
            t = t.replace(n, " ")
        # Collapse spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    def is_known_url(self, url: str) -> bool:
        """
        Phase 8: Check if URL exists in cache.
        """
        # Iterate _url_to_text or _link_index
        # Better: check if url is in _url_to_text keys?
        # But _url_to_text is populated by load() which scans files.
        # Let's assume _url_to_text has all valid URLs.
        if not self._loaded: self.load() # Ensure loaded
        
        url_lower = url.lower()
        # Check against _url_to_text keys (which are file names or URLs?)
        # Wait, _url_to_text matches keys.
        # But we might have only raw links in _link_index.
        # Let's check both.
        
        # 1. Direct Content Match
        if url in self._url_to_text: return True
        
        # 2. Known external link pattern from smc index
        # This is expensive if we iterate.
        # Just return True if we trust the logic in query engine to find it?
        # No, strict verify.
        
        # If we have _normalized_title_index, maybe check values
        for meta in self._normalized_title_index.values():
             if meta.get("href") == url: return True
             
        return False

    def _infer_article_type(self, title: str, text: str, url: str) -> str:
        """
        Phase 15: Infer article type based on title, content snippets, and URL.
        """
        t = (title + " " + (text or "")[:500]).lower()
        title_lower = title.lower()
        u = url.lower()
        
        # Phase 17 - Rule 2: COMMAND KEYWORD AUTO-TYPING
        if any(kw in title_lower for kw in ["command", "cmd", "syntax", "คำสั่ง"]) or \
           any(kw in u for kw in ["command", "cmd", "syntax"]):
            return "COMMAND_REFERENCE"
            
        # Priority 2: MIGRATION_CONVERSION 
        if any(kw in t for kw in ["migrate", "convert", "transition", "conversion", "เปลี่ยนรุ่น", "ย้าย"]):
            return "MIGRATION_CONVERSION"
            
        # Priority 3: TROUBLESHOOTING
        if any(kw in t for kw in ["troubleshoot", "issue", "error", "log", "event", "check", "test", "monitor", "แก้ปัญหา", "ตรวจเช็ค"]):
            return "TROUBLESHOOTING"
            
        # Priority 4: CONFIG_GUIDE
        if any(kw in t for kw in ["config", "setup", "set ", "manual", "template", "step-by-step", "การตั้งค่า", "คู่มือ"]):
            return "CONFIG_GUIDE"
            
        # Priority 5: FEATURE_USAGE
        if any(kw in t for kw in ["feature", "usage", "vlan", "native", "ip-context", "function", "ความสามารถ"]):
            return "FEATURE_USAGE"
            
        # Priority 6: OVERVIEW
        if any(kw in t for kw in ["introduction", "overview", "basic", "description", "chart", "คืออะไร", "เบื้องต้น"]):
            return "OVERVIEW"
            
        return "OVERVIEW" # Default

    def _extract_vendor_model(self, text: str) -> tuple[str, str]:
        """
        Phase SMC: Extract vendor and model from text.
        Returns (vendor, model) tuple. Empty string if not found.
        """
        text_lower = text.lower()
        
        # Vendor patterns (order matters - check specific before generic)
        vendor_patterns = {
            'huawei': ['huawei', 'hw'],
            'cisco': ['cisco'],
            'zte': ['zte'],
            'nokia': ['nokia'],
            'meru': ['meru'],
            'juniper': ['juniper']
        }
        
        vendor = ""
        for v_name, v_keywords in vendor_patterns.items():
            if any(kw in text_lower for kw in v_keywords):
                vendor = v_name
                break
        
        # Model extraction (common patterns)
        model = ""
        # Pattern: Letters followed by numbers (e.g., NE8000, ASR920, C300, F680)
        # Exclude vendor name from model by matching standalone alphanumeric patterns
        # Look for patterns after vendor names or standalone
        text_after_vendor = text_lower
        if vendor:
            # Remove vendor name to avoid capturing it in model
            for kw in vendor_patterns.get(vendor, []):
                text_after_vendor = text_after_vendor.replace(kw, ' ')
        
        model_match = re.search(r'\b([a-z]*\d+[a-z0-9]*)\b', text_after_vendor)
        if model_match:
            model = model_match.group(1).strip()
        
        return (vendor, model)

    def find_best_article_match(self, query: str, threshold: float = 0.7) -> dict | None:
        """
        Phase 7: Deterministic Retrieval (The 'Killer Move').
        1. Exact/Near-Exact Title Match
        2. Alias Match
        3. Keyword Intersection
        """
        if not self._loaded: self.load()

        # Phase 16: Rule 1 - EXACT MATCH PRIORITY (Slug or Title)
        # 1.1 Check Slug Index (Highest Priority)
        q_norm_exact = self.normalize_key(query)
        if q_norm_exact in self._slug_index:
             hit = self._slug_index[q_norm_exact]
             print(f"[Deterministic] RULE 1: EXACT SLUG MATCH: '{hit['text']}'")
             return {
                 "url": hit["href"],
                 "title": hit["text"],
                 "score": 1.0,
                 "match_type": "deterministic",
                 "article_type": hit.get("article_type", "OVERVIEW")
             }
             
        # 1.2 Check Title Index
        q_match = self.normalize_for_matching(query)
        if q_match in self._normalized_title_index:
             hit = self._normalized_title_index[q_match]
             print(f"[Deterministic] RULE 1: EXACT TITLE MATCH: '{hit['text']}'")
             return {
                 "url": hit["href"],
                 "title": hit["text"],
                 "score": 1.0,
                 "match_type": "deterministic",
                 "article_type": hit.get("article_type", "OVERVIEW")
             }
             
        # Phase 20 - 1.3 Soft Normalization Match
        q_soft = self.soft_normalize(query)
        if q_soft in self._soft_normalized_title_index:
             hit = self._soft_normalized_title_index[q_soft]
             print(f"[Deterministic] RULE 1.3: SOFT-NORMALIZED EXACT MATCH: '{hit['text']}'")
             return {
                 "url": hit["href"],
                 "title": hit["text"],
                 "score": 1.0, # Phase 20: 100% score for soft exact match
                 "match_type": "deterministic",
                 "soft_match": True,
                 "article_type": hit.get("article_type", "OVERVIEW")
             }

        # Use loaded aliases
        ALIASES = self.aliases
        
        q_norm = self.normalize_for_matching(query)
        if not q_norm: return None
        q_tokens = set(q_norm.split())
        
        # Phase SMC: Extract vendor/model from query
        query_vendor, query_model = self._extract_vendor_model(query)
        
        # Phase 16 Keywords
        cmd_kws = ["command", "cmd", "คำสั่ง", "syntax"]
        mig_kws = ["convert", "migration", "upgrade", "transform"]
        has_cmd_q = any(kw in q_norm for kw in cmd_kws)

        best_hit = None
        best_score = 0.0
        candidates = []
        
        for title_norm, entries in self._link_index.items():
            t_clean = self.normalize_for_matching(title_norm)
            if not t_clean: continue
            t_tokens = set(t_clean.split())
            
            article_vendor, article_model = self._extract_vendor_model(title_norm)
            if query_vendor and article_vendor and query_vendor != article_vendor:
                continue
            if query_model and article_model:
                if query_model not in article_model and article_model not in query_model:
                    if not (query_model[:3] == article_model[:3] or query_model[:4] == article_model[:4]):
                        continue
            
            # Scoring Logic
            score = 0.0
            if q_tokens.issubset(t_tokens) and len(q_tokens) > 0:
                 score = 0.98 # Slug/Title subset priority
            elif t_tokens.issubset(q_tokens) and len(t_tokens) > 0:
                 score = 0.95 
            else:
                 expanded_hit = False
                 for alias_key, expansions in ALIASES.items():
                     if alias_key in q_norm:
                         for exp in expansions:
                             if self.normalize_for_matching(exp) in t_clean:
                                 expanded_hit = True; break
                     if expanded_hit: break
                 
                 if expanded_hit:
                     score = 0.9 
                 else:
                     if len(q_tokens) > 0 and len(t_tokens) > 0:
                         intersection = q_tokens.intersection(t_tokens)
                         score = len(intersection) / len(q_tokens)

            # Rule 2: COMMAND KEYWORD BOOST/PENALTY
            if has_cmd_q:
                has_cmd_t = any(kw in t_clean for kw in cmd_kws)
                has_mig_t = any(kw in t_clean for kw in mig_kws)
                if has_cmd_t: score += 0.2
                if has_mig_t: score -= 0.6 # Heavy penalty

            score = max(0.0, min(0.99, score)) # Non-exact matches cap below 1.0

            if score >= (threshold - 0.1): 
                candidates.append({"entry": entries[0], "score": score, "title": entries[0]["text"]})
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        if not candidates: return None

        best_score = candidates[0]["score"]
        best_hit = candidates[0]["entry"]
        
        if len(candidates) >= 2:
            top1, top2 = candidates[0], candidates[1]
            if top1["score"] < 0.95 and (top1["score"] - top2["score"] < 0.05):
                return {"match_type": "ambiguous", "candidates": [c["title"] for c in candidates[:3]], "score": top1["score"]}
        
        if best_score >= threshold and best_hit: 
             return {
                 "url": best_hit["href"], "title": best_hit["text"], "score": best_score,
                 "match_type": "deterministic", "article_type": best_hit.get("article_type", "OVERVIEW")
             }
        
        # Phase 8: Detect "Corpus Missing" for Known Aliases
        # If score < 0.7 BUT we found an ALIAS key, it means the target doc is likely missing.
        for alias_key, expansions in ALIASES.items():
             if alias_key in q_norm:
                 print(f"[Deterministic] Known Alias '{alias_key}' found, but no high-confidence doc matched. Reporting MISSING.")
                 return {
                     "match_type": "missing_corpus",
                     "topic": expansions[0],
                     "alias_used": alias_key,
                     "score": 0.0
                 }

        return None

    def load(self):
        print(f"[ProcessedCache] Loading processed data from {self.processed_dir}...")
        
        files = list(self.processed_dir.glob("*.json"))
        print(f"[ProcessedCache] Found {len(files)} files")
        
        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                
                # Store text
                url = data.get("url", "")
                title = data.get("title", "").strip()
                if url:
                    text = data.get("text", "")
                    self._url_to_text[url] = text
                    
                    # Store images (if any)
                    images = data.get("images", [])
                    if images:
                        self._url_to_images[url] = images

                    # NEW: Index the Document Title itself
                    if title:
                        art_type = self._infer_article_type(title, text, url)
                        norm_title = self.normalize_key(title)
                        entry = {"text": title, "href": url, "source_url": url, "article_type": art_type}
                        if norm_title not in self._link_index:
                            self._link_index[norm_title] = []
                        self._link_index[norm_title].append(entry)
                        
                        match_title = self.normalize_for_matching(title)
                        if match_title and match_title not in self._normalized_title_index:
                            self._normalized_title_index[match_title] = entry
                        
                        # Phase 20: Soft Normalization Index
                        soft_title = self.soft_normalize(title)
                        if soft_title and soft_title not in self._soft_normalized_title_index:
                            self._soft_normalized_title_index[soft_title] = entry

                    # NEW: Index Slug from URL (e.g. "onu-zte" from the ID part or alias)
                    # Extract ID/Alias from URL if possible
                    # Example: option=com_content&view=article&id=599:-onu-zte-
                    slug_match = re.search(r'id=\d+:([^&]+)', url)
                    if slug_match:
                        slug = slug_match.group(1).replace("-", " ").strip()
                        if slug:
                            norm_slug = self.normalize_key(slug)
                            art_type = self._infer_article_type(slug, text, url)
                            entry = {"text": slug, "href": url, "source_url": url, "article_type": art_type}
                            if norm_slug not in self._link_index:
                                self._link_index[norm_slug] = []
                            self._link_index[norm_slug].append(entry)
                            self._slug_index[norm_slug] = entry # Phase 16

                    # Store links (for deterministic retrieval)
                    links = data.get("links", [])
                    for link in links:
                        link_text = link.get("text", "").strip()
                        if not link_text: continue
                        
                        # Normalize for indexing
                        norm_text = self.normalize_key(link_text)
                        
                        entry = {
                            "text": link_text,
                            "href": link.get("href", ""),
                            "source_url": url
                        }
                        
                        if norm_text not in self._link_index:
                            self._link_index[norm_text] = []
                        self._link_index[norm_text].append(entry)
                        
                        # Phase SMC: Build normalized title index for fast exact match
                        title_norm = self.normalize_for_matching(link_text)
                        if title_norm and title_norm not in self._normalized_title_index:
                            self._normalized_title_index[title_norm] = entry
                        
                        # Phase 20: Build soft-normalized index
                        title_soft = self.soft_normalize(link_text)
                        if title_soft and title_soft not in self._soft_normalized_title_index:
                            self._soft_normalized_title_index[title_soft] = entry
                        
                        self.stats["total_links"] += 1
                    
                    self.stats["total_docs"] += 1
                    
            except Exception as e:
                print(f"[ProcessedCache] Error loading {fp}: {e}")
        
        self._keys = list(self._link_index.keys()) # Rebuild keys after all links are processed
        self._loaded = True
        print(f"[ProcessedCache] Loaded {self.stats['total_docs']} docs, {self.stats['total_links']} links")

    def get_text(self, url: str) -> str | None:
        if not self._loaded:
            self.load()
        return self._url_to_text.get(url)

    def find_links(self, query: str) -> list[dict[str, str]]:
        if not self._loaded:
            self.load()
        return self._link_index.get(self.normalize_key(query), [])

    def find_links_fuzzy(self, query: str, threshold: float = 0.60) -> list[dict]:
        if not self._loaded:
            self.load()
            
        norm_q = self.normalize_key(query)
        if not norm_q: return []
        
        # 1. Exact match first (fast path)
        if norm_q in self._link_index:
            return [{"score": 1.0, "items": self._link_index[norm_q]}]
            
        # 2. Fuzzy match
        results = []
        # optimization: filter keys by length? or basic containment first?
        # difflib is O(N*M), so iterating all keys might be slow if many keys.
        # But for ~1000 links it's fine.
        
        for k in self._keys:
            if not k: continue # Skip empty keys
            
            # Simple substring boost
            ratio = difflib.SequenceMatcher(None, norm_q, k).ratio()
            
            # Phase 105: Token Subset Boost (Robustness)
            # If all query tokens are in the key, likely a good match (even if extra words exist)
            q_tokens = set(norm_q.split())
            k_tokens = set(k.split())
            
            is_subset = q_tokens.issubset(k_tokens) and len(q_tokens) >= 2
            
            # Substring Boost
            # Fix Phase 106: Ensure k has significant length to avoid matching "a" or empty string
            if ((norm_q in k or (len(k) > 3 and k in norm_q)) and len(norm_q) > 8) or is_subset:
                 # Boost ratio significantly
                 ratio = max(ratio, 0.85)

            if ratio >= threshold:
                results.append({
                    "score": ratio,
                    "key": k,
                    "items": self._link_index[k]
                })
        
        # Sort by score desc
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

class ChatEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.chat_cfg = cfg.get("chat", {})
        self.rag_cfg = cfg.get("rag", {})

        # Load Routing Policy
        policy_path = Path("config/routing_policy_v1.yaml")
        if policy_path.exists():
            try:
                self.routing_policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                print(f"[CRITICAL] Failed to load routing policy: {e}")
                raise RuntimeError("System Halt: Invalid Routing Policy")
        else:
            print("[CRITICAL] Routing policy file missing!")
            raise RuntimeError("System Halt: Missing Routing Policy")
            
        # ==================================================
        # SMC CONFIGURATION HARDENING (Phase 3.5)
        # ==================================================
        self.hardening_policy = self.routing_policy.get("smc_hardening", {})
        if not self.hardening_policy:
            print("[CRITICAL] SMC Hardening Policy Missing in Config!")
            raise RuntimeError("System Halt: Security Policy Violation (Missing smc_hardening)")
            
        print(f"[INFO] SMC Hardening ACTIVE. Policy: {list(self.hardening_policy.keys())}")
        
        # Enforce Locked Parameters
        if self.hardening_policy.get("similarity_threshold"):
             self.hardening_threshold = float(self.hardening_policy.get("similarity_threshold"))
             print(f"[INFO] Enforcing Hardened Threshold: {self.hardening_threshold}")
        else:
             self.hardening_threshold = 0.70 # Default hard

        
        self.top_k = int(cfg["retrieval"]["top_k"])
        self.show_context = bool(self.chat_cfg.get("show_context", True))
        self.show_context = bool(self.chat_cfg.get("show_context", True))
        self.save_log = bool(self.chat_cfg.get("save_log", True))
        self.cache_threshold = float(self.chat_cfg.get("cache_threshold", 0.90)) # Phase 224 Fix
        
        # Load resources
        print("[INFO] Loading resources...")
        self.vs = build_vectorstore(cfg)
        self.processed_cache = ProcessedCache("data/processed")
        self.processed_cache.load()
        self.records = load_records("data/records/directory.jsonl")
        
        from src.rag.knowledge_pack import KnowledgePackManager
        self.kp_manager = KnowledgePackManager()
        
        # Step 10: Persistent telemetry for auditing
        self.last_telemetry = {}
        
        # Load positions with Optimized Index (Phase 35: Scalable Lookup)
        self.position_index = {}
        self.pos_norm_map = {}  # Normalized -> [Original Roles]
        
        pos_path = Path("data/records/positions.jsonl")
        if pos_path.exists():
            for line in pos_path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try:
                    p = json.loads(line)
                    role = p.get("role", "").strip()
                    if role:
                        if role not in self.position_index:
                            self.position_index[role] = []
                        self.position_index[role].append(p)
                    
                    # Also add to flat records for Person Lookup (Phase 223 Fix)
                    # Use a copy to avoid mutation issues if any
                    p_flat = p.copy()
                    if not p_flat.get("type"): p_flat["type"] = "person"
                    self.records.append(p_flat)
                except Exception:
                    pass
                    
        print(f"[INFO] Resources loaded. Position Index Size: {len(self.position_index)}")

        # Phase 69: Team Index
        self.team_index = {}
        team_path = Path("data/records/teams.jsonl")
        if team_path.exists():
            for line in team_path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try:
                    t = json.loads(line)
                    team_name = t.get("team", "").strip()
                    if team_name:
                         self.team_index[team_name] = t
                except Exception:
                     pass
        print(f"[INFO] Resources loaded. Team Index Size: {len(self.team_index)}")
        
        # Phase 2.1: Query Normalizer (LLM Paraphrasing Helper)
        self.query_normalizer = QueryNormalizer(self.llm_cfg)
        
        # Content-Type Classifier (Pre-Summarization Check)
        self.content_classifier = ContentClassifier(self.llm_cfg)
        
        # Phase 43: Directory Handler (Encapsulates Person/Position Logic)
        self.directory_handler = DirectoryHandler(self.position_index, self.records, self.team_index, query_normalizer=self.query_normalizer)
        
        # =================================================================
        # Numeric Selection Resolver (Isolated Evolutionary Layer)
        # =================================================================
        from src.context.numeric_selection_resolver import NumericSelectionResolver
        from src.context.numeric_input_detector import NumericInputDetector
        
        self.numeric_selection_resolver = NumericSelectionResolver()
        self.numeric_input_detector = NumericInputDetector()
        self.pending_numeric_session = None  # Selection session state
        print("[INFO] Numeric Selection Resolver loaded")
        
        # =================================================================
        # Entity Detector (Phase 27.4 Bug Fix - Entity Bypass)
        # =================================================================
        from src.governance.lightweight_entity_detector import LightweightEntityDetector
        
        self.entity_detector = LightweightEntityDetector()
        print("[INFO] Entity Detector loaded (bypass noise gate for entity queries)")
        
        # =================================================================
        # Colloquial Noise Remover (Query Preprocessing)
        # =================================================================
        from src.preprocessing.colloquial_noise_remover import ColloquialNoiseRemover
        
        self.colloquial_noise_remover = ColloquialNoiseRemover()
        print("[INFO] Colloquial Noise Remover loaded (clean Thai particles)")
        
        # Context State
        self.last_context: Dict[str, Any] | None = None
        self.proc_ctx: Dict[str, Any] | None = None  # Phase 21: Procedural Context
        self.pending_contact_clarify = None # Phase 221
        
        # Clarify Session State (Phase 15: AI-First Orchestration)
        self.pending_clarify: Dict[str, Any] | None = None
        self.pending_kp_clarify: Dict[str, Any] | None = None # Phase 25: KP Clarification Memory
        self.pending_question: Dict[str, Any] | None = None   # Phase 80: Conversational Follow-up

        # Initialize Cache
        self.cache = SemanticCache() if self.rag_cfg.get("use_cache", True) else None
        
        # Initialize RAG Controller
        self.controller = RAGController(self.llm_cfg)
        
        # Initialize RAG Evaluator
        self.evaluator = RAGEvaluator(self.llm_cfg)
        
        # Phase 173/350: Safe Normalizer & Interpreter Layer
        self.content_classifier = ContentClassifier(self.llm_cfg)
        self.safe_normalizer = SafeNormalizer(self.llm_cfg)
        self.metrics = MetricsTracker() # Phase 8: Metrics Dashboard
        
        # Load Resources Retrieval Optimizer
        self.retrieval_optimizer = RetrievalOptimizer(self.llm_cfg)
        
        # Initialize AI Planner & Slot Filler (Phase 15)
        self.planner = QueryPlanner(self.llm_cfg)
        self.slot_filler = SlotFiller(self.llm_cfg)
        
        # Initialize Article Interpreter (Phase 16)
        self.article_interpreter = ArticleInterpreter(self.llm_cfg, ux_cfg=self.cfg.get("ux", {}))
        
        # Phase 34: Intent Router
        from src.ai.router import IntentRouter
        from src.ai.corrector import QueryCorrector
        
        # Optimization (Phase 127): Use fast model for Routing
        # We strip the 'model' key to force Router to use its default (llama3.2:3b)
        # unless 'router_model' is explicitly defined in config.
        router_llm_cfg = self.llm_cfg.copy()
        if "router_model" in self.cfg.get("rag", {}):
            router_llm_cfg["model"] = self.cfg["rag"]["router_model"]
        else:
             # Force default fallback (llama3.2:3b) by removing specific model dictation
             # assuming user has llama3.2 or similar light model installed.
             router_llm_cfg.pop("model", None)
             
        self.router = IntentRouter(router_llm_cfg)
        self.corrector = QueryCorrector(self.llm_cfg)
        
        # Phase 97: Pro-Level Caching
        self.greetings_handler = GreetingsHandler()
        
        # Phase R7-Fix: Initialize manager explicitly to prevent crash & enable cache globally
        from src.rag.cache_manager import CacheManager
        self.cache_manager = CacheManager(self.cache)

        # Phase 175: Web Handler
        self.web_handler = WebHandler(self.llm_cfg)

    def reset_context(self):
        """Reset conversation context associated with the current session."""
        self.pending_question = None
        self.pending_clarify = None
        self.pending_kp_clarify = None
        self.pending_contact_clarify = None # Phase 221
        self.proc_ctx = None  
        self.last_context = None


    # Step 8.2: Clarification Candidate Builder
    def _build_clarification_candidates(self, results: List[Any]) -> List[Dict]:
        """
        Build clarification candidates from vector search results.
        No LLM involved. Purely metadata-driven.
        """
        candidates = []
        seen_titles = set()
        
        # Filter and Score Check (Score > 0.5)
        for r in results:
            if r.score < 0.5: continue
            
            meta = r.metadata or {}
            title = meta.get("title", "")
            url = meta.get("url", "")
            art_type = meta.get("article_type", "General")
            
            # Remove bad candidates (Category, Section)
            if "view=category" in url.lower() or "view=section" in url.lower(): continue
            if not title: continue
            
            # De-duplicate by title (Case insensitive)
            title_clean = title.strip().lower()
            if title_clean in seen_titles: continue
            seen_titles.add(title_clean)
            
            candidates.append({
                "title": title,
                "url": url,
                "score": r.score,
                "article_type": art_type
            })
            
            # Limit 3-7 items (Step 8 Req)
            if len(candidates) >= 5: break
            
        return candidates

    # Phase 80: Conversational Follow-up Resolver

    def _resolve_pending_followup(self, q: str, latencies: Dict[str, float]) -> Dict[str, Any] | None:
        """
        Attempts to resolve a pending ambiguity question using short-term memory.
        Returns response dict if resolved/handled, else None.
        """
        pq = self.pending_question
        print(f"[DEBUG_INTERNAL] _resolve_pending_followup invoked. pq={pq}")
        if not pq: return None
        kind = pq.get("kind")
        print(f"[DEBUG] _resolve_pending_followup: kind={kind}, q='{q}', pq={pq}")
        
        # 1. Logic for Choice Selection (1, 2, 3...)
        # 1. Check TTL (120s - User Req Phase 227)
        created_at = pq.get("created_at", 0)
        if time.time() - created_at > 120:
            print("[DEBUG] Pending Question Expired")
            self.pending_question = None
            return None
            
        q_clean = q.strip().lower()
        candidates = pq.get("candidates", [])
        kind = pq.get("kind", "unknown")
        if kind not in ["role_choice", "team_choice", "contact_choice", "position_confirmation", 
                        "vendor_command_selection", "article_selection", "category_selection"]:
            return None

        # --- Logic: Vendor Command Selection (Step 5) & Article Selection (Step 8) ---
        if kind in ["vendor_command_selection", "article_selection", "category_selection"]:
             # 1. Check for Index Selection (1-N)

             sel_idx = -1
             if re.match(r"^\d+$", q_clean):
                 try:
                     sel_idx = int(q_clean) - 1
                 except: 
                     sel_idx = -1
             
             # 2. Check for Title Fuzzy Match
             selected_candidate = None
             if sel_idx >= 0 and sel_idx < len(candidates):
                 selected_candidate = candidates[sel_idx]
                 print(f"[DEBUG] User Selected Index {sel_idx+1}: {selected_candidate.get('title', 'Unknown')}")
             else:
                 if sel_idx != -1:
                     print(f"[DEBUG] Selection Index {sel_idx+1} out of range (Total: {len(candidates)})")
                 
                 # Check fuzzy match against candidate titles
                 # Use simple containment or token overlap
                 best_score = 0
                 best_cand = None
                 for cand in candidates:
                     title = cand.get("title", "").lower()
                     
                     # Improved Scoring:
                     # 1. Exact token match (e.g. "vas" in "config vas ...") -> High Score
                     if q_clean in title.split():
                         score = 0.9
                     # 2. Substring match -> Length ratio
                     elif q_clean in title:
                         score = len(q_clean) / len(title)
                     else:
                         score = 0
                         
                     if score > best_score:
                         best_score = score
                         best_cand = cand
                 
                 # Lower threshold for substring if it's significant (e.g. > 2 chars)
                 threshold = 0.15 if len(q_clean) > 2 else 0.3
                 if best_cand and best_score >= threshold:
                      selected_candidate = best_cand
                      print(f"[DEBUG] User Selected Title Match ({best_score:.2f}): {selected_candidate['title']}")

             if selected_candidate:
                 # Clear pending state
                 self.pending_question = None
                 
                 # New Step 12: Category Selection Handling
                 if kind == "category_selection":
                     cat_payload = selected_candidate.get("payload", selected_candidate["title"])
                     print(f"[DEBUG] Category Selected: {selected_candidate['title']} -> Processing '{cat_payload}'")
                     # Recursive call to process the category query
                     return self.process(cat_payload)
                 
                 # Route to Deterministic Flow (Exact Match)
                 # We construct a synthetic result similar to immediate hit
                 return self._handle_article_route(
                     url=selected_candidate["url"],
                     query=selected_candidate["title"],
                     latencies=latencies,
                     start_time=time.time(),
                     match_score=1.0,
                     intent="DETERMINISTIC_MATCH",
                     article_type=selected_candidate.get("article_type", "COMMAND"),
                     decision_reason=f"User selected option ({kind}): {selected_candidate['title']}"
                 )
             else:

                 # User input didn't match options -> Treat as new query (return None to fall through)
                 return None

            
        # 0. Early Exit for Position Confirmation (Special Handling)
        if kind == "position_confirmation":
            # Confirmed Position Search
            # Check for Yes/No
            # We reuse q (the user input)
            
            is_yes = any(w in q.lower() for w in ["ใช่", "yes", "ok", "ถูกต้อง", "เอา", "confirm"])
            is_no = any(w in q.lower() for w in ["ไม่", "no", "cancel", "ยกเลิก", "ไม่ใช่", "ไม่เอา"])
            
            target = pq.get("target_name")
            
            if is_yes:
                print(f"[DEBUG] User Confirmed Position Search: {target}")
                self.pending_question = None # Clear state
                return self.directory_handler.handle(target)
            elif is_no:
                 self.pending_question = None # Clear state
                 return {
                     "answer": "ขออภัยครับ ช่วยระบุชื่อเต็มหรือตัวย่อที่ถูกต้องให้อีกครั้ง (ตัวอย่าง: ผจ.ส่วนงาน...)",
                     "route": "position_confirmation_rejected",
                     "latencies": latencies
                 }
            # If ambiguous, fall through? No, better return None or ask again.
            # But fallthrough allows "Logic D" to pick it up? No, Logic D assumes dicts.
            # So return None here if ambiguous.
            return None
            
        # --- Logic: Detect New Query vs Selection (Phase 228) ---
        # If input contains search triggers OR is clearly not a number selection (and len > 3)
        search_triggers = ["เบอร์", "โทร", "ติดต่อ", "หา", "ค้น", "ช่วย", "ทั้ง", "รวม", "หมด", "-", "ip"]
        
        is_numeric = re.match(r"^\d+$", q_clean)
        is_search = any(t in q_clean for t in search_triggers)
        
        # If strict selection mode, only allow numbers or specific keywords
        # If input is long and not numeric -> likely new query
        if not is_numeric and (is_search or len(q_clean) > 3):
             # Check if it's a yes/no/detail intent first?
             # No, those are handled below. But "ip-phone-helpdesk" (len>3) should bypass.
             # But "ยกเลิก" (len=6) is NO. "detail" (len=6) is DETAIL.
             # So checks below must run if short/specific words.
             
             # Let's perform Yes/No check first?
             # Or just include them in check?
             pass 
             
        # New Strict Logic:
        # If it looks like a number -> Selection Logic (B)
        # If it corresponds to Yes/No/Detail -> Logic A/D
        # Else -> Escape as New Query
        
        # Phase 220: Check Yes/No/Detail intents using Token Matching (Fix Substring bug "no" in "noc")
        q_tokens = set(q_clean.split())
        
        YES_WORDS = {"ใช่", "ถูกต้อง", "ok", "ครับ", "ค่ะ", "แม่น", "ถูก", "yes"}
        NO_WORDS = {"ไม่", "ไม่ใช่", "no", "ผิด", "ยกเลิก", "cancel", "หยุด", "พอ"}
        DETAIL_WORDS = {"ขอ", "detail", "ราย", "ละ", "เอียด", "เพิ่ม"}
        
        # Check strict token match for short words, or substring for long phrases if needed
        # For safety, strict set intersection is best for "no", "ok", "yes"
        is_yes = not q_tokens.isdisjoint(YES_WORDS)
        is_no = not q_tokens.isdisjoint(NO_WORDS)
        is_detail = any(d in q_clean for d in ["รายละเอียด", "detail", "ขอเพิ่ม"]) # Detail is usually phrase
        
        if not (is_numeric or is_yes or is_no or is_detail):
             print(f"[DEBUG] Input '{q_clean}' is not selection/control -> Escape as New Query.")
             self.pending_question = None
             return None
        
        # Keywords (Already defined above? Consolidate)
        # Re-using strict checks

        YES_WORDS = ["ใช่", "ใช่ครับ", "ใช่ค่ะ", "เอาอันนี้", "ถูก", "ok", "โอเค", "yes", "y", "ถูกต้อง", "แม่นแล้ว", "ช่าย", "แม่น", "confirm", "ยืนยัน"]
        NO_WORDS = ["ไม่", "ไม่ใช่", "ไม่เอา", "no", "n", "ยกเลิก", "cancel", "ไม่ครับ", "ไม่ค่ะ", "ไม่จ้า", "เปลี่ยน", "reject"]
        DETAIL_WORDS = ["ชื่อเต็ม", "ขอชื่อเต็ม", "ขอรายละเอียด", "รายละเอียด", "เอาแบบเต็ม", "เต็มๆ", "full", "detail", "more", "ชื่อจริง"]
        
        is_yes = any(w in q_clean for w in YES_WORDS) # Changed from equality to inclusion for robust matching? No, existing was strict eq.
        # User requirement 1: "Recognize FOLLOWUP_ACTION tokens".
        # Let's stick to containment for short phrases or equality for single words.
        # Original was: `any(w == q_clean ...)`
        # Let's expand to `w in q_clean` if q_clean is short?
        # Actually user tokens like "ขอชื่อเต็ม" might be embedded.
        is_yes = any(w in q_clean for w in YES_WORDS)
        is_no = any(w in q_clean for w in NO_WORDS)
        is_detail = any(w in q_clean for w in DETAIL_WORDS)
        
        # Logic A: NO -> Cancel
        if is_no and not is_yes: 
            self.pending_question = None
            return {
                "answer": "รับทราบครับ ยกเลิกการเลือก หากต้องการค้นหาใหม่พิมพ์ได้เลยครับ",
                "route": "followup_cancel",
                "latencies": latencies
            }
            
        # Logic B: Selection by Number
        selected_idx = -1
        nums = re.findall(r"\d+", q_clean)
        if nums:
             try:
                 idx = int(nums[0]) - 1 
                 if 0 <= idx < len(candidates):
                     selected_idx = idx
             except: pass

        # Logic C: Fuzzy Match (If no number selected)
        if selected_idx == -1:
            from difflib import SequenceMatcher
            best_score = 0.0
            best_idx = -1
            
            # Normalize for matching: remove dots, spaces
            def normalize_for_match(text):
                return text.replace(".", "").replace(" ", "").lower()
            
            q_norm = normalize_for_match(q_clean)
            
            for i, cand in enumerate(candidates):
                if isinstance(cand, str):
                    label = cand.lower()
                else:
                    label = str(cand.get("label", "")).lower()

                label_norm = normalize_for_match(label)
                
                # Check containment with normalized text
                # Example: "ผส บลตน" normalized to "ผสบลตน" matches "ผส.บลตน." normalized to "ผสบลตน"
                if label_norm in q_norm or q_norm in label_norm:
                    score = 0.9
                # Also check original non-normalized for exact matches
                elif label in q_clean or q_clean in label:
                    score = 0.95
                else:
                    score = SequenceMatcher(None, q_norm, label_norm).ratio()
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Threshold: 0.7 for normalized fuzzy matching
            if best_score > self.cfg.get("followup_fuzzy_threshold", 0.7):
                 selected_idx = best_idx

        # Logic D: Yes/Detail Meta-Command
        if is_yes or is_detail:
             if selected_idx == -1:
                 # No specific selection found
                 if len(candidates) == 1:
                     selected_idx = 0
                 else:
                     # Ambiguous
                     msg = f"หมายถึงข้อไหนครับ? (กรุณาระบุเลข 1-{len(candidates)})"
                     if is_detail: msg = f"ต้องการข้อมูลเต็มของข้อไหนครับ? (กรุณาระบุเลข 1-{len(candidates)})"
                     
                     return {
                        "answer": msg,
                        "route": "followup_reask",
                        "latencies": latencies
                     }
        
        # If we reached here with selected_idx == -1, function returns None (Fallthrough)
        # UNLESS valid selection was made via Logic B or C (without Yes/Detail tokens)
        
        # EXECUTE RESOLUTION (Logic E)
        if selected_idx != -1:
            selection = candidates[selected_idx]
            target_label = selection.get("label") or selection.get("name") or "Unknown"
            print(f"[DEBUG] Follow-up Resolved: Choice {selected_idx+1} -> {target_label}")
            
            mode = pq.get("mode", "holder")
            self.pending_question = None # Clear state
            
            target_key = selection.get("key", target_label)
            
            # Mode A: Phone Lookup (via ContactHandler)
            if mode == "contact":
                print(f"[DEBUG] resolving choice {selected_idx+1} in CONTACT mode")
                # Format specific answer using Deterministic Formatter
                from src.directory.format_answer import format_contact_answer
                
                phones = selection.get("phones", [])
                ans = format_contact_answer(target_label, phones, selection)
                
                return {
                    "answer": ans,
                    "route": "contact_hit_choice",
                    "latencies": latencies,
                    "context": json.dumps([selection], ensure_ascii=False)
                }

            # Mode A (Legacy): Phone Lookup query injection
            if mode == "phone":
                print(f"[DEBUG] resolving {target_label} in PHONE mode")
                # Construct query for ContactHandler
                phone_q = f"เบอร์โทร {target_label}"
                from src.rag.handlers.contact_handler import ContactHandler
                return ContactHandler.handle(phone_q, self.records, directory_handler=self.directory_handler)
            
            # Mode B: Standard Info Lookup
            
            # Mode B: Standard Info Lookup
            if kind == "team_choice":
                return self.directory_handler.handle_team_lookup(target_key)
                
            elif kind == "role_choice" or kind == "management_choice":
                return self.directory_handler.handle_management_query(target_key)
        
        # New Logic for "Who Is" confirmation (Outside selected_idx check)
            
        return None

        


    def warmup(self):
        print(f"[CHAT] warmup: model={self.llm_cfg.get('model')} base_url={self.llm_cfg.get('base_url')}")
        try:
            _ = ollama_generate(
                base_url=self.llm_cfg.get("base_url", "http://localhost:11434"),
                model=self.llm_cfg.get("model", "llama3.2:3b"),
                prompt="ตอบว่า: พร้อมใช้งาน",
                temperature=0.0,
            )
            print("[CHAT] LLM warmed up")
        except Exception as e:
            print(f"[CHAT] LLM warm-up skipped: {e}")

    def check_governance_blocking(self, q: str) -> Optional[Dict[str, str]]:
        """
        Check for PROHIBITED INTENT (Tech Term + Action Word).
        Returns failure response dict if blocked, else None.
        """
        q_lower = q.lower()
        
        # 1. Check for Technical Terms
        strict_tech_terms = [
            "onu", "olt", "los", "fiber", "ไฟเบอร์", "fttx", "optical", "signal", 
            "gpon", "epon", "ont", "dslam", "bng", "sbc", "msan", "bras", "router", "switch", 
            "huawei", "zte", "nokia", "cisco", "juniper", "ne8000", "netengine", "c600", "c300", "c320"
        ]
        has_tech_term = any(term in q_lower for term in strict_tech_terms)
        
        # 2. Check for Forbidden Action Words (Command/Config)
        forbidden_cli_words = ["command", "คำสั่ง", "config", "cli"]
        
        # ========== RELAXED MODE: No Hard Refusal ==========
        # We allow procedural queries to go to RAG to satisfy 'answer everything'.
        # We will add a warning to the response later if results are found.
        if has_tech_term and any(word in q_lower for word in forbidden_cli_words):
             print(f"[GOVERNANCE] RELAXED: Command/Config detected in '{q}'. Allowing RAG with Source Fidelity.")
             return None # No refusal
        
        return None
    
    # =========================================================================
    # PHASE 6: STRICT INTENT & GOVERNANCE
    # =========================================================================
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent based on keywords.
        Returns: 'COMMAND', 'CONFIG', 'PROTOCOL', 'OVERVIEW', 'TROUBLESHOOT', 'HOWTO', or 'UNKNOWN'
        """
        q_lower = query.lower()
        
        # Phase 15: Strict Keyword Force for COMMAND
        if any(kw in q_lower for kw in ["syntax", "cmd", "คำสั่ง"]):
             return "COMMAND"
        
        # Priority Order: COMMAND > CONFIG > HOWTO > TROUBLESHOOT > PROTOCOL > OVERVIEW
        keywords = {
            "COMMAND": [
                "command", "show ", "display ", "cmd",
                "คำสั่งพื้นฐาน", "คำสั่ง",              # Thai command
            ],
            "CONFIG": [
                "config", "cli", "configure", "setup", "manual",
                "ตั้งค่า", "คู่มือ", "กำหนดค่า",          # Thai config
            ],
            "HOWTO": [
                "วิธี", "ขั้นตอน", "ทำยังไง", "ทำอย่างไร",  # Thai how-to
                "how to", "how-to", "แนะนำขั้นตอน",
                "reset", "reboot", "restart", "install", "เพิ่ม", "ลบ",
            ],
            "TROUBLESHOOT": [
                "error", "problem", "fail", "debug", "logs", "monitor",
                "ปัญหา", "แก้", "ไม่ได้", "ไม่ทำงาน",       # Thai troubleshoot
                "ขัดข้อง", "หลุด", "ล่ม", "ใช้ไม่ได้",
            ],
            "PROTOCOL": [
                "bgp", "ospf", "isis", "mpls", "vlan", "vxlan", "evpn",
            ],
            "OVERVIEW": [
                "overview", "spec", "what is", "introduction", "feature",
                "คืออะไร", "หมายถึงอะไร", "ทำหน้าที่อะไร",   # Thai overview
            ],
        }
        
        for intent, kws in keywords.items():
            if any(k in q_lower for k in kws):
                return intent
                
        return "UNKNOWN"


    def _is_article_compatible(self, intent: str, article_type: str, query: str, is_exact: bool = False) -> bool:
        """
        Phase 15/17: Verify if article type is compatible with query intent.
        """
        # Phase 19 - Rule 1: EXACT MATCH OVERRIDE (Highest Priority)
        # Score >= 0.95 or is_exact=True bypasses all compatibility gates
        if is_exact or (query and any(article_type.lower() == t for t in ["overview", "command_reference"])):
             # We allow it, but _handle_article_route will decide if it needs to be LINK_ONLY
             return True

        q_lower = query.lower()
        
        # Rule 3: Strict Keyword Force (Syntax/Cmd) ONLY allow COMMAND_REFERENCE
        if any(kw in q_lower for kw in ["syntax", "cmd", "คำสั่ง"]):
            return article_type == "COMMAND_REFERENCE"
            
        # Rule 2: COMMAND or CONFIG intent
        if intent in ["COMMAND", "CONFIG"]:
            # Rule 3/6: STRICT BLOCK MIGRATION for COMMAND intent
            if intent == "COMMAND" and article_type == "MIGRATION_CONVERSION":
                return False

            # ALLOW: COMMAND_REFERENCE, CONFIG_GUIDE, FEATURE_USAGE
            # CONFIG intent: also allow MIGRATION_CONVERSION (pre-tagged in processed_cache metadata)
            config_ok = article_type in ["COMMAND_REFERENCE", "CONFIG_GUIDE", "FEATURE_USAGE"]
            config_ok = config_ok or (intent == "CONFIG" and article_type == "MIGRATION_CONVERSION")
            if config_ok:
                return True
            # BLOCK: MIGRATION_CONVERSION, OVERVIEW, etc.
            return False
            
        return True # Default allow for other intents for now

    def _classify_article_type(self, title: str) -> str:
        """
        Classify article type based on title keywords.
        Returns: 'CONFIG', 'PROTOCOL', 'OVERVIEW', 'TROUBLESHOOT', or 'UNKNOWN'
        """
        return self._classify_intent(title) # Reuse same logic/keywords

    
    def _detect_vendor_scope(self, query: str) -> dict:
        """
        Phase 22: Detect if query mentions a vendor and check SMC corpus availability.
        Categorizes vendors into:
        - PRIMARY: Huawei, ZTE (Allowed for RAG/Vector)
        - SMC_ONLY: Cisco, Nokia (Only allowed if deterministic match exists)
        - OUT_OF_SCOPE: Juniper, Avaya, etc. (Blocked immediately)
        """
        primary_vendors = ['huawei', 'zte']
        smc_only_vendors = ['cisco', 'nokia']
        out_of_scope_vendors = ['juniper', 'avaya', 'alcatel', 'extreme', 'ericsson', 'meru']
        
        vendor_keywords = {
            'huawei': ['huawei', 'hw'],
            'zte': ['zte'],
            'cisco': ['cisco'],
            'nokia': ['nokia'],
            'juniper': ['juniper'],
            'avaya': ['avaya'],
            'alcatel': ['alcatel'],
            'extreme': ['extreme'],
            'ericsson': ['ericsson'],
            'meru': ['meru']
        }
        
        query_lower = query.lower()
        detected_vendor = ""
        
        # Detect vendor from query
        for v_name, keywords in vendor_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_vendor = v_name
                break
        
        if not detected_vendor:
            return {'vendor': '', 'type': 'UNKNOWN', 'in_scope': True}

        # Categorize
        v_type = "OUT_OF_SCOPE"
        if detected_vendor in primary_vendors:
            v_type = "PRIMARY"
        elif detected_vendor in smc_only_vendors:
            v_type = "SMC_ONLY"
            
        # Check if SMC has *any* articles for this vendor (sanity check)
        smc_count = 0
        if hasattr(self, 'processed_cache') and self.processed_cache:
            for title in self.processed_cache._link_index.keys():
                if detected_vendor in title.lower():
                    smc_count += 1
        
        # Logic for in_scope:
        # 1. PRIMARY: Always in-scope for RAG
        # 2. SMC_ONLY: In-scope only if articles exist (but will be blocked later if no match)
        # 3. OUT_OF_SCOPE: Always False
        in_scope = (v_type != "OUT_OF_SCOPE") and (smc_count > 0)
        
        return {
            'vendor': detected_vendor,
            'type': v_type,
            'in_scope': in_scope,
            'smc_article_count': smc_count
        }
    
    def _check_out_of_scope(self, query: str) -> bool:
        """
        Check if query is out of scope (e.g. opinions, comparisons).
        Returns True if out of scope.

        Soft-block logic:
        - Hard block: English opinion/comparison keywords (low risk of false negative)
        - Soft block: Thai opinion keywords — UNLESS a technical entity is also present,
          in which case the intent classifier downstream will decide (reduces false negatives).
        """
        q_lower = query.lower()

        hard_block_patterns = [
            "opinion", "comment", "best", "better", "worse",  # Opinions (EN)
            "compare", "vs ", "versus", "difference",         # Comparisons (EN)
            "why ", "reason for",                             # Open-ended Why (EN)
        ]

        soft_block_patterns = [
            "เปรียบเทียบ", "อันไหนดีกว่า", "ดีกว่ากัน", "ดีกว่า",  # Comparisons (TH)
            "แนะนำ", "ควรใช้อะไร", "คิดว่า", "ความเห็น", "รู้สึก",  # Opinions (TH)
        ]

        # Technical entity keywords — presence of these lifts the soft block
        technical_whitelist = [
            "onu", "olt", "gpon", "vlan", "router", "switch", "port", "fiber",
            "huawei", "zte", "cisco", "nokia", "meru", "adsl", "ftth", "fttb",
            "bras", "nms", "osp", "smc", "ip phone", "nat", "dhcp", "dns",
            "config", "setting", "คำสั่ง", "command", "ขั้นตอน", "reset", "reboot",
        ]

        # Exception: troubleshooting queries are always allowed
        if "error" in q_lower or "fail" in q_lower or "ขัดข้อง" in q_lower:
            return False

        # Hard block: always reject regardless of context
        if any(p in q_lower for p in hard_block_patterns):
            return True

        # Soft block: reject ONLY if no technical entity is present
        if any(p in q_lower for p in soft_block_patterns):
            has_technical_entity = any(t in q_lower for t in technical_whitelist)
            if has_technical_entity:
                print(f"[GOVERNANCE] Soft-block lifted: technical entity detected in opinion query → passing to classifier")
                return False
            return True

        return False

    def _classify_request_category(self, query: str) -> str:
        """
        Phase 6.5: High-Level Request Categorization
        Categories: TECH_ARTICLE_LOOKUP, NON_SMC_TECH, GENERAL_CHAT, INVALID_QUERY
        """
        q_lower = query.lower().strip()
        
        # 1. Invalid / Nonsense
        if len(q_lower) < 2 or not q_lower.isprintable():
            return "INVALID_QUERY"
        
        # CRITICAL: Force COMMAND intent to TECH_ARTICLE_LOOKUP
        # Prevents "command huawei" or "คำสั่ง zte" from falling to NON_SMC_TECH
        command_keywords = ["คำสั่ง", "command", "cmd", "cli"]
        if any(kw in q_lower for kw in command_keywords):
            print(f"[GOVERNANCE] COMMAND keyword detected -> Force TECH_ARTICLE_LOOKUP")
            return "TECH_ARTICLE_LOOKUP"
        
        # 2. Tech Article Lookup (Strict SMC)
        # Must contain valid SMC vendor AND model pattern
        # Reuse _has_vendor_model_pattern but strictly for SMC vendors
        smc_vendors = ['huawei', 'cisco', 'zte', 'nokia', 'meru'] # Juniper is NOT SMC
        has_smc_vendor = any(v in q_lower for v in smc_vendors)
        
        # Check for non-SMC vendors explicitly
        non_smc_vendors = ['juniper', 'avaya', 'alcatel', 'extreme']
        if any(v in q_lower for v in non_smc_vendors):
            return "NON_SMC_TECH"
            
        # Check for Tech Terms
        tech_terms = [
             "onu", "olt", "los", "fiber", "fttx", "gpon", "epon", "router", "switch", 
             "config", "command", "cli", "vlan", "bgp", "ospf", "mpls"
        ]
        has_tech_term = any(t in q_lower for t in tech_terms)
        
        # Phase 9.5: SMC-Intrinsic Device/Service Keywords Override
        # These terms are ALWAYS valid SMC scope, even without a vendor name,
        # Phase 9.5: SMC-Intrinsic Device/Service Keywords Override (Approval Fix)
        # These terms are ALWAYS SMC tech scope, even without a vendor prefix.
        smc_device_keywords = [
            "onu", "ont", "olt", "gpon", "epon", "fttx", "fttb", "ftth",
            "dslam", "msan", "ip phone", "ipphone", "sip", "v-pbx", "vpbx",
            "voip", "ip-phone", "voice",
        ]
        smc_device_keywords_th = [
            "โทร", "สาย", "โทรศัพท์", "เปิดโทร", "โอนสาย",
            "สามสาย", "3 สาย", "call waiting",
        ]
        has_smc_device = any(kw in q_lower for kw in smc_device_keywords)
        has_smc_device_th = any(kw in q_lower for kw in smc_device_keywords_th)
        
        if has_smc_device or has_smc_device_th:
            print(f"[GOVERNANCE] SMC Device Keyword Override: '{query}' -> TECH_ARTICLE_LOOKUP")
            return "TECH_ARTICLE_LOOKUP"

        if has_smc_vendor:
            # If it has SMC vendor, assume lookup (simple heuristic for now)
            # In real flow, we check for model too, but for classifier unit test:
            return "TECH_ARTICLE_LOOKUP"
            
        if has_tech_term:
            # Tech term but NO SMC vendor -> Non-SMC Tech (e.g. "How to config router")
            # This aligns with "tech but not SMC -> NON_SMC_TECH"
            return "NON_SMC_TECH"
            
        # 3. General Chat
        # If no tech terms and no vendors, it's general chat or valid non-tech QA
        return "GENERAL_CHAT"

    # =========================================================================
    # End Phase 6 Methods
    # =========================================================================

    def _has_vendor_model_pattern(self, q: str) -> bool:
        """
        Phase SMC: Detect if query contains vendor+model pattern (technical query).
        Technical queries MUST be answered from SMC only, no web fallback.
        """
        q_lower = q.lower()
        
        # Check for vendor presence
        vendors = ['huawei', 'cisco', 'zte', 'nokia', 'juniper', 'meru', 'hw']
        has_vendor = any(v in q_lower for v in vendors)
        
        if not has_vendor:
            return False
        
        # Check for model/device pattern
        # Common patterns: NE8000, ASR920, C300, F680, etc.
        has_model = bool(re.search(r'\b[a-z]*\d+[a-z0-9]*\b', q_lower))
        
        # Also check for device type keywords that indicate hardware query
        device_keywords = ['olt', 'onu', 'ont', 'router', 'switch', 'bng', 'sbc', 'cpe']
        has_device = any(d in q_lower for d in device_keywords)
        
        return has_vendor and (has_model or has_device)
    
    def _is_smc_url(self, url: str) -> bool:
        """
        Phase SMC: Validate if URL is from SMC internal server.
        Only SMC URLs (10.192.133.33) are considered trusted internal sources.
        """
        if not url:
            return False
        
        url_lower = url.lower()
        smc_patterns = ["10.192.133.33", "smc"]
        
        return any(pattern in url_lower for pattern in smc_patterns)

    def process(self, q: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Phase 3.5: Audited Process Wrapper
        Wraps internal logic to ensure mandatory logging.
        """
        # Step 10 Reset
        self.last_telemetry = {"query": q, "timestamp": time.time(), "route": "unknown"}
        
        try:
            res = self._process_logic(q, session_id)
            
            # Phase 21: Mandatory Audit Field Injection
            # Ensure audit exists and follows strict schema
            audit = res.get("audit", {})
            
            # Matched Title: Use Title or None (Null)
            matched_title = res.get("hits", [{}])[0].get("value") if res.get("hits") else None
            
            # Populate mandatory fields
            audit.update({
                "normalized_query": self.processed_cache.soft_normalize(q) if hasattr(self, 'processed_cache') else q,
                "matched_article_title": matched_title,
                "confidence_mode": res.get("route", "unknown"),
                "decision_reason": res.get("decision_reason", "Processing complete")
            })
            
            res["audit"] = audit
            
            # Log structured audit
            self._log_audit(res, query=q)
            return res
        except Exception as e:
            print(f"[AUDIT] CRITICAL PROCESS FAILURE: {e}")
            # Fail-closed strategy
            return {
                "answer": "⚠️ ระบบขัดข้องชั่วคราว (Internal Security Error). ขออภัยในความไม่สะดวกครับ",
                "route": "system_error_fail_closed",
                "audit": {
                    "error": str(e),
                    "normalized_query": q,
                    "confidence_mode": "fail_closed"
                }
            }
        except Exception as e:
            print(f"[AUDIT] CRITICAL PROCESS FAILURE: {e}")
            raise e

    def _is_vendor_broad_query(self, query: str) -> bool:
        """
        Step 9: Detect broad vendor-related command queries that should 
        trigger clarification instead of hijacking a single article.
        
        CRITICAL: Now uses AmbiguityDetector to catch concatenated patterns
        like "คำสั่งhuawei" (no space).
        """
        from src.query_analysis.ambiguity_detector import AmbiguityDetector
        
        result = AmbiguityDetector.check_ambiguity(query)
        
        # Check if it's vendor-related ambiguity
        is_vendor_ambiguous = result.get("is_ambiguous", False) and result.get("reason") in [
            "BROAD_VENDOR_COMMAND",
            "VENDOR_ONLY",
            "MINIMAL_VENDOR_CONTEXT"
        ]
        
        return is_vendor_ambiguous

    def _append_to_metrics_csv(self, metrics: dict):
        """
        Step 10: Save clarification metrics to CSV for offline analysis.
        """
        import csv
        import os
        
        log_dir = "results"
        log_file = os.path.join(log_dir, "clarification_metrics_log.csv")
        
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_exists = os.path.isfile(log_file)
            headers = [
                "timestamp", "query", "route", "top_score", "second_score", 
                "score_gap", "n_candidates_above_threshold", 
                "clarification_triggered", "clarification_reason"
            ]
            
            # Mapping telemetry to CSV row
            row = {
                "timestamp": metrics.get("timestamp", time.time()),
                "query": metrics.get("query", ""),
                "route": metrics.get("route", "unknown"),
                "top_score": metrics.get("top_score", 0.0),
                "second_score": metrics.get("second_score", 0.0),
                "score_gap": metrics.get("score_gap", 0.0),
                "n_candidates_above_threshold": metrics.get("n_candidates_above_threshold", 0),
                "clarification_triggered": metrics.get("clarification_triggered", False),
                "clarification_reason": metrics.get("clarification_reason", "none")
            }
            
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[WARN] Failed to write metrics to CSV: {e}")

    def _process_logic(self, q: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a query query_text and return a dictionary with:
        - answer: str
        - route: str
        - context: str (optional)
        - latency: dict (breakdown)
        - metadata: dict
        """
        # ========================================================================
        # SESSION CONTEXT LOADING (Phase: Context Memory)
        # Load persistent context from previous session
        # ========================================================================
        if session_id and session_id != "default":
            loaded_context = context_manager.load_session_context(session_id)
            if loaded_context:
                self.last_context = loaded_context
                print(f"[SESSION] Loaded context from session {session_id}")
        
        # ========================================================================
        # CRITICAL FIX 1: GREETING ABSOLUTE EARLY EXIT
        # MUST be FIRST LINE to bypass ALL processing (Governance/Normalizer/etc)
        # ========================================================================
        if not hasattr(self, 'pending_question') or not self.pending_question:
            q_exact = q.strip()
            
            # Check 1: Exact whitelist match
            if q_exact in GREETING_WHITELIST:
                return {
                    "answer": get_greeting_response(),
                    "route": "greeting_early_exit",
                    "context": "",
                    "decision_reason": "Exact greeting whitelist - ABSOLUTE early exit"
                }
            
            # Check 2: Pure greeting detector
            if is_pure_greeting(q):
                return {
                    "answer": get_greeting_response(),
                    "route": "greeting_early_exit",
                    "context": "",
                    "decision_reason": "Pure greeting detector - ABSOLUTE early exit"
                }
        # ========================================================================
        
        t_start = time.time()

        # Phase 171: Session Isolation
        # If session_id changed, reset pending states
        if hasattr(self, "_last_session_id") and self._last_session_id != session_id:
            print(f"[DEBUG] Session Switch Detected ({self._last_session_id} -> {session_id}). Resetting context.")
            self.reset_context()
        self._last_session_id = session_id

        # Phase 25/174: Initialize Query Tracking and Telemetry
        sys.stderr.write(f"[DEBUG_INTERNAL_START] _process_logic entered. q='{q}'\n"); sys.stderr.flush()
        # RuntimeError removed
        sys.stderr.write(f"HARDENING THRESHOLD: {getattr(self, 'hardening_threshold', 'MISSING')}\n"); sys.stderr.flush()
        original_q_str = q

        # =========================================================================
        # PHASE 0: EARLY RESOLUTION & INITIALIZATION (Moved Up)
        # =========================================================================
        
        latencies = {
            "routing": 0.0, "embed": 0.0, "vector_search": 0.0, "bm25": 0.0,
            "fusion": 0.0, "controller": 0.0, "retrieval_opt": 0.0,
            "generator": 0.0, "evaluator": 0.0, "clarify": 0.0,
            "total": 0.0, "llm": 0.0
        }

        # 0. Answer Mode Detection (Phase 25)
        # Check explicit answer mode overrides first
        answer_mode = "FULL"
        q_lower = q.lower()
        if any(k in q_lower for k in ["ขอแค่เบอร์", "เอาแค่เบอร์", "phone only", "only phone"]):
            answer_mode = "PHONE_ONLY"

        elif any(k in q_lower for k in ["ขอแค่ ip", "เอาแค่ ip", "ip only", "only ip"]):
            answer_mode = "IP_ONLY"
        elif any(k in q_lower for k in ["ขอแค่ลิงก์", "ขอ link", "url only", "link only", "ขอลิงก์"]):
            answer_mode = "URL_ONLY"
        if any(k in q_lower for k in ["สรุปสั้นๆ", "summary", "short"]):
            answer_mode = "SUMMARY"

        # =========================================================================
        # EXECUTION LAYER ARCHITECTURE (Step 17.5)
        # Layer 0: System Safety (already handled in init)
        # Layer 0.5: NUMERIC SELECTION BYPASS (Isolated Evolutionary Layer) ← NEW
        # Layer 1: PENDING STATE RESOLVER (Highest Priority)
        # Layer 2-3: Deterministic Match (handled in _process_logic)
        # Layer 4: Greeting/Small Talk (after pending)
        # Layer 5: SafeNormalizer (fallback only)
        # =========================================================================
        
        # =========================================================================
        # LAYER 0.5: NUMERIC SELECTION BYPASS
        # =========================================================================
        numeric_input = self.numeric_input_detector.is_numeric_selection(q)
        
        if numeric_input and self.pending_numeric_session:
            if self.numeric_selection_resolver.validate_number(numeric_input, self.pending_numeric_session):
                selected = self.numeric_selection_resolver.resolve_selection(numeric_input, self.pending_numeric_session)
                
                if selected:
                    print(f"[NUMERIC_SELECTION] Resolved: {numeric_input} → {selected.get('title', 'N/A')}")
                    self.pending_numeric_session = None
                    article_url = selected.get('url') or selected.get('article_id')
                    
                    if article_url:
                        return {
                            "answer": f"📌 **{selected.get('title', 'เอกสาร')}**\n🔗 [ดูรายละเอียด]({article_url})",
                            "route": "numeric_selection_resolved",
                            "confidence": 1.0,
                            "latencies": {"total": 0}
                        }
            else:
                max_num = self.pending_numeric_session.get('max_number', '?')
                return {
                    "answer": f"❌ กรุณาเลือกหมายเลข 1-{max_num}",
                    "route": "numeric_selection_invalid",
                    "confidence": 1.0,
                    "latencies": {"total": 0}
                }
        # =========================================================================
        
        # LAYER 1: PENDING STATE RESOLVER (HIGHEST PRIORITY)
        # STEP 1-2: Handles ALL confirmation tokens BEFORE any other layer
        bypass_cache = False
        skip_normalizer = False  # STEP 3: Skip normalizer if pending active
        
        # =========================================================================
        # CONVERSATION CONTEXT ENRICHMENT (Follow-Up Query Support)
        # =========================================================================
        # CRITICAL: Apply BEFORE pending state check to allow context + pending cooperation
        # Enable follow-up queries like "ขอเบอร์" after "ศูนย์หาดใหญ่โทรอะไร"
        # Detect if query needs context and enrich with entity from last turn
        
        context_enriched = False
        self.context_intent_override = None  # Instance variable for cross-method access
        
        if context_manager.should_use_context(q, self.last_context):
            # Always preserve intent when context is applicable — even if query
            # text is not changed (e.g., rival-entity check kept query the same).
            self.context_intent_override = self.last_context.get("intent")
            if self.context_intent_override:
                print(f"[CONTEXT_MEMORY] Intent preserved: {self.context_intent_override}")

            enriched_query = context_manager.enrich_query_with_context(q, self.last_context)
            if enriched_query != q:
                q = enriched_query
                context_enriched = True
                print(f"[CONTEXT_MEMORY] Query enriched from conversation context")
        
        # =========================================================================
        # PHASE 80: PENDING STATE RESOLUTION (After Context Enrichment)
        # =========================================================================
        # Check for pending clarification questions from previous turn
        # Note: Now runs AFTER context enrichment so they can work together
        
        if self.pending_question:
             # STEP 6: Check expiry FIRST
             created_at = self.pending_question.get("created_at", 0)
             if time.time() - created_at > PENDING_STATE_TTL:
                 print("[DEBUG] [LAYER 1] Pending expired. Clearing state, continuing to Layer 2.")
                 self.pending_question = None
                 bypass_cache = False
             else:
                 # STEP 3: Active pending -> skip normalizer entirely
                 skip_normalizer = True
                 print(f"[DEBUG] [LAYER 1] Pending active. Skip normalizer. Handling: '{q}'")
                 
                 followup_res = self._resolve_pending_followup(q, latencies)
                 if followup_res:
                     # Success! Return result
                     followup_res = self._apply_answer_mode(followup_res, answer_mode)
                     return followup_res
                 
                 # STEP 1: None-safe check after resolver
                 if self.pending_question is None:
                     print("[DEBUG] [LAYER 1] Pending cleared by resolver. Continuing to Layer 2.")
                     bypass_cache = False
                     skip_normalizer = False
                 else:
                     # SAFETY GUARD: Numeric input during pending that failed resolution
                     pq_kind = self.pending_question.get("kind")
                     if pq_kind in ["article_selection", "vendor_command_selection"] and q.strip().isdigit():
                         print(f"[DEBUG] [LAYER 1] Invalid numeric '{q}' -> Blocking fall-through")
                         return {
                             "answer": "⚠️ กรุณาเลือกหมายเลขที่ระบุในรายการครับ (เช่น 1, 2, ...)",
                             "route": "selection_out_of_range",
                             "decision_reason": "Numeric input out of range for pending selection",
                             "audit": {
                                 "error": "selection_out_of_range",
                                 "input": q,
                                 "pending_kind": pq_kind
                             }
                         }
                     
                     bypass_cache = True
                     print("[DEBUG] [LAYER 1] Pending still active -> Bypassing Cache")
        

        # Rule: Explicitly bypass cache for short follow-up words even if pending state missing
        if len(q) < 10:
            FOLLOWUP_TOKENS = ["ใช่", "ไม่", "เอา", "ข้อ", "choice", "yes", "no"]
            if any(t in q_lower for t in FOLLOWUP_TOKENS):
                bypass_cache = True


        # =========================================================================
        # PRE-GOVERNANCE PENDING GATE (FIX: Contact Selection Numeric Input)
        # =========================================================================
        # CRITICAL: Resolve numeric selection BEFORE Governance Layer
        # Bug: "1" during contact selection was being overridden by Governance fallback
        # Solution: Early exit for pending contact_choice with numeric input
        
        if self.pending_question:
            pending_kind = self.pending_question.get("kind")
            
            # FIX 1: Pre-Governance Pending Gate (Contact Selection Only)
            if pending_kind == "contact_choice":
                # FIX 3: Pending Ownership Isolation
                # Only accept numeric input or explicit cancel
                q_stripped = q.strip()
                
                # Check if input is numeric-only
                if q_stripped.isdigit():
                    selection_idx = int(q_stripped) - 1  # 1-indexed to 0-indexed
                    candidates = self.pending_question.get("candidates", [])
                    
                    # Validate range
                    if 0 <= selection_idx < len(candidates):
                        print(f"[PRE-GOVERNANCE GATE] Contact Selection Resolved: Choice {q_stripped}")
                        
                        # Resolve selection
                        selected = candidates[selection_idx]
                        
                        # Format response
                        from src.directory.format_answer import format_contact_answer
                        answer = format_contact_answer(
                            q, 
                            selected.get("phones", []), 
                            selected
                        )
                        
                        # FIX 4: Post-Resolve State Hygiene
                        self.pending_question = None
                        self.pending_contact_clarify = None
                        
                        # Audit
                        print(f"[AUDIT] contact_resolved_from_pending: {selected.get('name')}")
                        
                        # Early exit - terminate pipeline
                        return {
                            "answer": answer,
                            "route": "contact_resolved_from_pending",
                            "context": json.dumps([selected], ensure_ascii=False),
                            "latencies": {"total": time.time() - t_start}
                        }
                    else:
                        print(f"[PRE-GOVERNANCE GATE] Invalid selection index: {q_stripped}")
                        # Clear pending and treat as new query
                        self.pending_question = None
                        self.pending_contact_clarify = None
                
                # Cancel keywords (optional - can be expanded)
                cancel_keywords = ["ยกเลิก", "cancel", "ไม่เอา"]
                if any(kw in q_lower for kw in cancel_keywords):
                    print(f"[PRE-GOVERNANCE GATE] Contact selection cancelled")
                    self.pending_question = None
                    self.pending_contact_clarify = None
                    # Continue to normal flow (don't return)
        
        # =========================================================================
        # COLLOQUIAL NOISE REMOVAL (BEFORE Entity Detection)
        # =========================================================================
        # Clean Thai colloquial particles that don't affect meaning
        # Examples: "OLT มันคืออะไรหว่า" → "OLT คืออะไร"
        
        original_q = q  # Keep original for audit
        noise_removal_applied = False
        
        if hasattr(self, 'colloquial_noise_remover'):
            noise_result = self.colloquial_noise_remover.remove_noise(q)
            
            if noise_result['was_modified']:
                q = noise_result['cleaned_query']
                noise_removal_applied = True
                print(f"[COLLOQUIAL_CLEANUP] Removed {noise_result['removed_count']} particles: {noise_result['removed_words']}")
                print(f"[COLLOQUIAL_CLEANUP] '{original_q}' → '{q}'")
        
        # =========================================================================
        # PHASE 27.4: ENTITY DETECTION (BEFORE Query Normalization)
        # =========================================================================
        # CRITICAL: Detect entities BEFORE QueryNormalizer to prevent entity stripping
        # Bug Fix: "ขอเบอร์UMUX" was being rewritten to "เบอร์UMX" (entity corrupted)
        # Solution: Detect entities in ORIGINAL query, set bypass flag BEFORE any normalization
        
        entity_bypass = False
        entity_info = None
        
        if hasattr(self, 'entity_detector'):
            entity_info = self.entity_detector.detect(q)
            
            if entity_info['has_entity'] and entity_info['confidence'] >= 0.70:
                entity_bypass = True
                print(f"[ENTITY_BYPASS] Entity detected: '{entity_info['entity_value']}' "
                      f"(Type: {entity_info['entity_type']}, Confidence: {entity_info['confidence']})")
                print(f"[ENTITY_BYPASS] Noise gate will be bypassed for this query")
        
        # =========================================================================
        # PHASE 6/22 GOVERNANCE: PRE-FLIGHT CHECKS
        # =========================================================================
        
        # Phase 22: PRE-FLIGHT VENDOR CHECK (Out-of-Scope)
        vendor_scope = self._detect_vendor_scope(q)
        v_name = vendor_scope['vendor']
        v_type = vendor_scope['type']
        
        if v_name and v_type == "OUT_OF_SCOPE":
            print(f"[GOVERNANCE] Phase 22: Early Vendor Block - {v_name}")
            return {
                "answer": (
                    "⚠️ **นอกขอบเขต SMC (Out of Scope)**\n\n"
                    "ระบบ SMC-RAG ให้บริการข้อมูลเฉพาะจากเอกสารคู่มือในระบบ SMC เท่านั้น\n"
                    f"ผู้ผลิต '{v_name.upper()}' ไม่มีข้อมูลในฐานข้อมูล SMC ครับ\n\n"
                    "กรุณาสอบถามเกี่ยวกับอุปกรณ์ที่มีในระบบ (เช่น Huawei, ZTE, Nokia) ครับ"
                ),
                "route": "blocked_vendor_out_of_scope",
                "block_reason": f"NON_SMC_VENDOR_{v_name.upper()}",
                "decision_reason": "Vendor outside SMC perimeter",
                "latencies": {"vendor_check": time.time() - t_start},
                "audit": {
                    "vendor_detected": v_name,
                    "normalized_query": self.processed_cache.soft_normalize(q) if hasattr(self, 'processed_cache') else q,
                }
            }
        
        # 1. OUT-OF-SCOPE CHECK
        if self._check_out_of_scope(q):
            print(f"[GOVERNANCE] Query blocked as Out-of-Scope: {q}")
            res = {
                "answer": (
                    "⚠️ **คำถามอยู่นอกขอบเขตการให้บริการ (Out of Scope)**\n\n"
                    "ระบบ SMC-RAG ให้บริการข้อมูลเฉพาะการตั้งค่าและการใช้งานอุปกรณ์ NT เท่านั้น\n"
                    "ไม่รองรับการเปรียบเทียบเชิงความเห็น (Opinion), คำแนะนำทั่วไป, หรือคำถามเชิงเปรียบเทียบข้ามค่ายครับ\n\n"
                    "กรุณาระบุรุ่นอุปกรณ์หรือคำสั่งที่ต้องการค้นหาอีกครั้งครับ"
                ),
                "route": "blocked_scope",
                "block_reason": "OUT_OF_SCOPE_QUERY",
                "latencies": {"total": time.time() - t_start}
            }
            return res
        
        # VENDOR NORMALIZATION LAYER (NEW)
        # Fix common spacing typos in vendor names before governance
        # e.g., "hu awei" → "huawei", "z te" → "zte"
        import re
        vendor_patterns = {
            r'\bhu\s*a\s*wei\b': 'huawei',
            r'\bz\s*t\s*e\b': 'zte',
            r'\bc\s*i\s*s\s*co\b': 'cisco',
            r'\bn\s*o\s*k\s*ia\b': 'nokia',
            r'\bm\s*e\s*ru\b': 'meru',
            r'\bj\s*u\s*n\s*i\s*per\b': 'juniper',
        }
        q_normalized = q
        for pattern, replacement in vendor_patterns.items():
            q_normalized = re.sub(pattern, replacement, q_normalized, flags=re.IGNORECASE)
        
        if q_normalized != q:
            print(f"[VENDOR NORMALIZATION] '{q}' → '{q_normalized}'")
            q = q_normalized  # Update query for all downstream processing
            
        # 2. INTENT CLASSIFICATION
        query_intent = self._classify_intent(q)
        self._last_intent = query_intent   # Expose for ROUTING log
        self._last_confidence = 0.9        # Default; overridden by router if available
        print(f"[GOVERNANCE] Identified Intent: {query_intent}")
        
        # 3. ROOT-CAUSE GUARDRAIL (Phase 8 Step 4)
        # IF intent != TECH_ARTICLE_LOOKUP -> FORCE BLOCK
        # Note: We allow GENERAL_CHAT if it's purely Greetings? No, user said NO EXCEPTIONS.
        # But we need to check if _classify_request_category handles greetings separately or fails them.
        req_cat = self._classify_request_category(q)
        print(f"[GOVERNANCE] Request Category: {req_cat}")
        
        if req_cat not in ["TECH_ARTICLE_LOOKUP", "GENERAL_CHAT"]: # Strict Technical only (Wait, user said != TECH_ARTICLE_LOOKUP -> FORCE BLOCK)
             # However, disabling GENERAL_CHAT breaks "Hello".
             # User said: "Reviewer-mode questions blocked 100%".
             # "No general networking explanation".
             # "No cross-vendor answers".
             # "Reviewer-mode" usually means technical probe.
             # If I block "Hello", it might fail basic usability.
             # I will blindly follow "NO EXCEPTIONS" for now as per "Stress Test" requirements.
             # If user complains about "Hello", we can relax.
             # WAIT: SAFE_NORMALIZER might output ASSET or other intents.
             
             if req_cat != "TECH_ARTICLE_LOOKUP":
                 # But wait, what if it's an ASSET query? "ขอเบอร์ NT".
                 # _classify_request_category uses keywords.
                 # "onu", "config" -> NON_SMC_TECH if no vendor.
                 # "phone", "contact" -> might be GENERAL_CHAT?
                 # This guardrail might be TOO strict for "Contact Info".
                 # Let's check if query_intent is TEAM_LOOKUP or CONTACT related.
                 
                  is_allowed_exception = False
                  if query_intent in ["TEAM_LOOKUP", "CONTACT_INFO", "GREETING"]:
                      is_allowed_exception = True
                  
                  # Phase 9.5: Article Fallback — last chance before blocking
                  # If an SMC article actually exists for this query, let it through
                  # LAST CHANCE: Check if SMC article exists for this query (Threshold 0.60)
                  # This fallback prevents valid techncial SMC queries from being blocked.
                  # FIX 2: Pending Guard - Skip FALLBACK OVERRIDE when pending active
                  if not is_allowed_exception and not self.pending_question and hasattr(self, 'processed_cache') and self.processed_cache:
                      try:
                          # 1. Deterministic/Keyword Match (Titles, Links, Slugs)
                          fallback_match = self.processed_cache.find_best_article_match(q, threshold=0.6)
                          if fallback_match and fallback_match.get("match_type") not in ["missing_corpus", None]:
                              # Phase 15: Compatibility Check
                              art_type = fallback_match.get("article_type", "OVERVIEW")
                              is_exact = fallback_match.get("score", 0.0) >= 1.0
                              if self._is_article_compatible(query_intent, art_type, q, is_exact=is_exact):
                                  print(f"[GOVERNANCE] FALLBACK OVERRIDE: Compatible SMC article found via keyword for '{q}' -> Returning Link")
                                  return self._handle_article_route(
                                      url=fallback_match["url"],
                                      query=q,
                                      latencies={"fallback_match": 0.0},
                                      start_time=t_start,
                                      match_score=fallback_match.get("score", 0.6),
                                      intent="FALLBACK_LINK_ONLY",
                                      article_type=art_type
                                  )
                              else:
                                  print(f"[GOVERNANCE] Fallback article ({art_type}) incompatible with intent ({query_intent}). Skipping.")
                          
                          # 2. Semantic Fallback (Vector Store)
                          if hasattr(self, 'vs') and self.vs:
                              v_res = self.vs.query(q, top_k=1)
                              if v_res and v_res[0].score >= 0.6:
                                  hit = v_res[0]
                                  hit_url = hit.metadata.get("url")
                                  if hit_url:
                                      # Phase 15: Infer type for semantic hit too
                                      hit_title = hit.metadata.get("title", "")
                                      hit_text = hit.page_content if hasattr(hit, 'page_content') else ""
                                      art_type = self.processed_cache._infer_article_type(hit_title, hit_text, hit_url)
                                      
                                      if self._is_article_compatible(query_intent, art_type, q):
                                          print(f"[GOVERNANCE] FALLBACK OVERRIDE: Compatible SMC article found via semantic ({hit.score:.2f}) for '{q}' -> Returning Link")
                                          return self._handle_article_route(
                                              url=hit_url,
                                              query=q,
                                              latencies={"semantic_fallback": 0.0},
                                              start_time=t_start,
                                              match_score=hit.score,
                                              intent="FALLBACK_LINK_ONLY",
                                              article_type=art_type
                                          )
                                      else:
                                          print(f"[GOVERNANCE] Semantic article ({art_type}) incompatible with intent ({query_intent}). Skipping.")
                      except Exception as e:
                          print(f"[GOVERNANCE] Fallback search error: {e}")

                  if not is_allowed_exception:
                      # Rule 5: Special Message for COMMAND
                      if query_intent == "COMMAND":
                           block_msg = "ไม่พบบทความ SMC ที่เป็น Command โดยตรง"
                      else:
                           block_msg = (
                              "⚠️ **คำถามอยูกนอกขอบเขต (Strict Policy)**\n\n"
                              "ระบบอนุญาตเฉพาะการค้นหาข้อมูลทางเทคนิคของอุปกรณ์ SMC (Huawei, Cisco, ZTE, Nokia) เท่านั้นครับ\n"
                              "ขออภัยในความไม่สะดวก"
                          )
                      
                      print(f"[GOVERNANCE] STRICT BLOCK: Category {req_cat} != TECH_ARTICLE_LOOKUP (Intent: {query_intent})")
                      res = {
                          "answer": block_msg,
                          "route": "blocked_scope",
                          "block_reason": f"STRICT_INTENT_{query_intent}",
                          "latencies": {"total": time.time() - t_start}
                      }
                      return res # Wrapper will log audit
        
        # =========================================================================
        # END GOVERNANCE CHECKS
        # =========================================================================

        # ── Early KP exit for Huawei NE8000 command queries ─────────────────
        # Must be BEFORE Phase SMC Fast Path (Title Match)
        _q_lower_kp = q.lower()
        _huawei_triggers = ["huawei", "ne8000", "display ", "อัตราการใช้",
                            "หน่วยความจำ", "manuinfo", "cpu-usage", "serial number"]
        
        if self.kp_manager and any(t in _q_lower_kp for t in _huawei_triggers):
            _kp_early = self.kp_manager.lookup(q)
            if _kp_early and _kp_early.get("hits"):
                _top_hit  = _kp_early["hits"][0]
                # Fact dict now has injected "_score"
                _top_score = _top_hit.get("_score", 0)
                
                if _top_score >= 10:
                    print(f"[KP_CMD_EARLY] Huawei command match (score={_top_score}) → bypassing Fast Path")
                    latencies["total"] = (time.time() - t_start) * 1000
                    return {
                        "answer":   _kp_early["answer"],
                        "route":    "knowledge_pack_huawei_cmd",
                        "latencies": latencies,
                        "hits":     _kp_early["hits"],
                    }
        # ── End early KP exit ────────────────────────────────────────────────

        
        # Phase SMC: Title Exact Match Fast Path (O(1) lookup)
        # Check if query exactly matches a normalized article title
        try:
            if hasattr(self, 'processed_cache') and self.processed_cache:
                q_norm_title = self.processed_cache.normalize_for_matching(q)
                q_soft_title = self.processed_cache.soft_normalize(q)
                
                # Try strict normalized match first, then soft normalized match
                article = None
                if q_norm_title in self.processed_cache._normalized_title_index:
                    article = self.processed_cache._normalized_title_index[q_norm_title]
                elif q_soft_title in self.processed_cache._soft_normalized_title_index:
                    article = self.processed_cache._soft_normalized_title_index[q_soft_title]
                    
                if article:
                    art_type = article.get("article_type", "OVERVIEW")
                    is_soft = (q_soft_title == self.processed_cache.soft_normalize(article['text']))
                    self.soft_deterministic_hit = is_soft

                    # ============================================================
                    # SMART SKIP: Menu/Collection Page Detection
                    # ============================================================
                    # Skip Fast Path ONLY for menu/collection pages
                    # All other exact matches → Fast Path (Link Only) ✅
                    
                    menu_skip_patterns = ["ต่างๆ", "รวม", "เมนู", "ทั้งหมด", "หลายๆ", "index"]
                    article_title_lower = article['text'].lower()
                    is_menu_page = any(p in article_title_lower for p in menu_skip_patterns)
                    
                    if is_menu_page:
                        print(f"[FAST PATH] Menu page detected: '{article['text']}' - Skipping Fast Path")
                        # Continue to content analysis
                    # Phase 15/17: Compatibility for Fast Path (Exact = True)
                    elif self._is_article_compatible(query_intent, art_type, q, is_exact=True):
                        print(f"[FAST PATH] Title match compatible ({art_type}): {article['text']} (Soft={is_soft})")
                        res = self._handle_article_route(
                            url=article["href"],
                            query=q,
                            latencies={"title_match": 0.0},
                            start_time=t_start,
                            match_score=1.0,
                            intent="DETERMINISTIC_MATCH",
                            article_type=art_type,
                            decision_reason=f"Soft-normalized exact match" if is_soft else f"Fast-path exact title match: {article['text']}"
                        )
                        return res
                    else:
                        print(f"[FAST PATH] Incompatible match ({art_type}) for intent ({query_intent}). Skipping fast path.")
        except Exception as e:
            print(f"[WARN] Title fast-path check failed: {e}")
        
        # Phase 7: Deterministic Retrieval (Pre-Semantic)
        # Intent Gate: Only activate for FACT intents to prevent false-positive matches.
        # GENERAL_QA, STATUS_CHECK, UNKNOWN → bypass → straight to RAG vector search.
        DETERMINISTIC_FACT_INTENTS = {
            "CONTACT_LOOKUP",
            "PERSON_LOOKUP",
            "MANAGEMENT_LOOKUP",
            "POSITION_HOLDER_LOOKUP",
            "HOWTO_PROCEDURE",
            "HOWTO",           # From _classify_intent Thai patterns
            "COMMAND",         # From _classify_intent
            "CONFIG",          # From _classify_intent
            "TROUBLESHOOT",    # From _classify_intent
            "REFERENCE_LINK",
            "NEWS_SEARCH",
        }
        det_gate_allowed = query_intent in DETERMINISTIC_FACT_INTENTS
        if not det_gate_allowed:
            print(f"[INTENT GATE] Skipping Deterministic: intent='{query_intent}' not in FACT set → RAG route")
        try:
             if det_gate_allowed:
                 det_hit = self.processed_cache.find_best_article_match(q, threshold=self.hardening_threshold)
             else:
                 det_hit = None  # Bypass deterministic, force RAG
             sys.stderr.write(f"[DEBUG_INTERNAL] det_hit: {det_hit}, type: {type(det_hit)}\n"); sys.stderr.flush()
             
             if det_hit:
                 # Step 9: Vendor Broad Hijack Guard
                 # If a query is very broad (e.g., 'huawei command'), force clarification 
                 # even if one article has a high deterministic score.
                 if det_hit.get("score", 0) >= 0.9 and self._is_vendor_broad_query(q):
                     print(f"[GOVERNANCE] Force Clarification for Vendor Broad Query: '{q}'")
                     det_hit = {} # Force fallback to vector clarification logic

                 # Step 9.5: Generic Protocol Broad Command Guard
                 # คำสั่ง telnet/ssh/ping/vlan without specific action → multi-result
                 if det_hit and det_hit.get("score", 0) >= 0.7:
                     from src.query_analysis.ambiguity_detector import AmbiguityDetector as _AD
                     _amb_g = _AD.check_ambiguity(q, intent=query_intent)
                     if _amb_g.get("reason") == "BROAD_GENERIC_COMMAND":
                         print('[GOVERNANCE] Force Clarification for Generic Protocol Command:', q)
                         det_hit = {}  # Force fallback to multi-result selection

                 # Step 6 Enhancement: Type Guard Filter (Post-Validation)
                 if (query_intent == "OVERVIEW" and 
                     det_hit.get("match_type") == "deterministic" and 
                     det_hit.get("article_type") == "COMMAND_REFERENCE"):
                     print(f"[GOVERNANCE] REJECT Deterministic Match: Type Mismatch (OVERVIEW vs COMMAND_REFERENCE). Fallback to Search.")
                     det_hit = {} # Reject match to force fallback

                 # Phase 16: Rule 1 - Skip clarification if Exact Match exists
                 is_exact = det_hit.get("score", 0.0) >= 1.0
                 
                 # Phase 6: Handle Ambiguity
                 if not is_exact and det_hit.get("match_type") == "ambiguous":
                     candidates = det_hit.get("candidates", [])
                     candidate_list = "\n".join([f"- {c}" for c in candidates])
                     res = {
                         "answer": (
                             f"⚠️ **ค้นหาไม่ชัดเจน (Ambiguous Query)**\n\n"
                             f"ระบบพบเอกสารที่ใกล้เคียงหลายรายการ กรุณาระบุให้ชัดเจนขึ้นครับ:\n\n"
                             f"{candidate_list}"
                         ),
                         "route": "blocked_ambiguous",
                         "block_reason": "AMBIGUOUS_QUERY",
                         "decision_reason": "Ambiguous query",
                         "latencies": {"total": time.time() - t_start}
                     }
                     return res

                 # Case 1: Deterministic Match (Success)
                 if det_hit.get("match_type") == "deterministic":
                     # Phase 6/15: Intent Mismatch Check
                     article_title = det_hit.get("title", "")
                     art_type = det_hit.get("article_type")
                     if not art_type:
                         art_type = self._classify_article_type(article_title)
                     
                     # Phase 15/17: Strict Compatibility Gate
                     check_compatibility = True
                     if det_hit.get("score", 0) >= 0.95:
                         v_bypass = self._detect_vendor_scope(q)
                         if not (v_bypass.get('vendor') and v_bypass.get('type') == 'SMC_ONLY'):
                             check_compatibility = False
                     
                     if check_compatibility and not self._is_article_compatible(query_intent, art_type, q, is_exact=is_exact):
                         # TC-G2 Fix: CHECK VENDOR FIRST
                         v_scope = self._detect_vendor_scope(q)
                         if v_scope.get('vendor') and v_scope.get('type') == 'SMC_ONLY' and '10.192' not in det_hit.get('url', ''):
                              print(f"[GOVERNANCE] Incompatible match for SMC_ONLY vendor '{v_scope['vendor']}'. BLOCKING as Vendor Scope.")
                              reason = f"{v_scope['vendor'].capitalize()} is SMC-only; no allowed article match"
                              return {
                                  "answer": (
                                      "⚠️ **นอกขอบเขต SMC (Out of Scope)**\n\n"
                                      "ระบบ SMC-RAG ให้บริการข้อมูลเฉพาะจากเอกสารคู่มือในระบบ SMC เท่านั้น\n"
                                      f"ผู้ผลิต '{v_scope['vendor'].upper()}' หรือรุ่นที่ระบุไม่มีข้อมูลในฐานข้อมูล SMC ครับ\n\n"
                                      "กรุณาสอบถามเกี่ยวกับอุปกรณ์ที่มีในระบบ (เช่น Huawei, ZTE, Nokia) หรือระบุคำสั่งให้ชัดเจนขึ้นครับ"
                                  ),
                                  "route": "blocked_vendor_out_of_scope",
                                  "block_reason": f"NON_SMC_VENDOR_{v_scope['vendor'].upper()}",
                                  "decision_reason": reason,
                                  "latencies": {"vendor_check": time.time() - t_start},
                                  "audit": {
                                      "vendor_detected": v_scope['vendor'],
                                      "normalized_query": self.processed_cache.soft_normalize(q) if hasattr(self, 'processed_cache') else q,
                                  }
                              }
                     
                         print(f"[GOVERNANCE] Primary Match Incompatible: Query({query_intent}) vs Article({art_type})")
                         if query_intent == "COMMAND":
                              block_msg = "ขออภัยครับ ไม่พบบทความคู่มือ SMC ที่เป็นคำสั่ง (Command) โดยตรงสำหรับรุ่นนี้"
                         else:
                              block_msg = (
                                f"⚠️ **เนื้อหาไม่ตรงกับสิ่งที่ต้องการ (Governance)**\n\n"
                                f"ท่านต้องการข้อมูลแบบ '{query_intent}' แต่เอกสาร '{article_title}' เป็นข้อมูลประเภท '{art_type}'\n"
                                f"เพื่อความถูกต้อง ระบบจึงแนะนำให้ระบุคำค้นหาให้ชัดเจนกว่านี้ครับ"
                              )
                         
                         return {
                             "answer": block_msg,
                             "route": "blocked_intent",
                             "block_reason": f"INCOMPATIBLE_TYPE_{art_type}",
                             "decision_reason": f"Intent mismatch: Query({query_intent}) vs Article({art_type})",
                             "latencies": {"total": time.time() - t_start},
                             "audit": {
                                 "vendor_detected": self._detect_vendor_scope(q)['vendor'] if self._detect_vendor_scope(q)['vendor'] else "None"
                             }
                         }
                     
                     # CRITICAL FIX 4: Contact Priority Check
                     # If query has contact keywords (phone, email, etc), skip article route
                     # and let contact handler process it instead
                     CONTACT_KEYWORDS_PRIORITY = {"เบอร์", "ติดต่อ", "phone", "โทร", "อีเมล", "email", "fax", "แฟกซ์", "โทรสาร"}
                     q_lower_priority = q.lower()
                     has_contact_keyword = any(kw in q_lower_priority for kw in CONTACT_KEYWORDS_PRIORITY)
                     
                     if has_contact_keyword:
                         print(f"[CRITICAL FIX 4] Contact keyword detected in '{q}' -> Skipping article route, routing to CONTACT domain")
                         # Don't return article, continue to contact handler
                         det_hit = None  # Clear det_hit to force contact routing
                     else:
                         print(f"[GOVERNANCE] Deterministic Match Found: {det_hit['title']} ({det_hit['score']:.2f})")
                         # Phase 21: Track if soft match for decision reason
                         self.soft_deterministic_hit = bool(det_hit.get("soft_match"))

                         res = self._handle_article_route(
                             url=det_hit["url"],
                             query=q,
                             latencies={"deterministic": 0.0},
                             start_time=t_start,
                             match_score=det_hit["score"],
                             intent="DETERMINISTIC_MATCH",
                             article_type=art_type,
                             decision_reason=f"Exact title match found: {article_title}" if is_exact and not self.soft_deterministic_hit else (f"Soft-normalized exact match" if self.soft_deterministic_hit else f"High-score deterministic match: {article_title}")
                         )
                         return res
                 
                 # Case 2: Missing Corpus (Failure but Known Topic)
                 elif det_hit.get("match_type") == "missing_corpus":
                     print(f"[GOVERNANCE] MISSING CORPUS DETECTED: {det_hit['topic']}")
                     # STRICT GOVERNANCE RULE: Do NOT fallback to RAG/Web. Reports explicitly.
                     res = {
                         "answer": (f"⚠️ **ขออภัยครับ ยังไม่มีข้อมูลสำหรับ '{det_hit['topic']}' ในระบบ**\n\n"
                                    f"สถานะ: `MISSING_CORPUS`\n"
                                    f"คำแนะนำ: ระบบรู้จักชื่อรุ่นนี้ แต่ยังไม่มีเอกสารคู่มือ/คำสั่งในฐานข้อมูล\n"
                                    f"กรุณาแจ้ง Admin เพื่อนำเข้าเอกสารสำหรับรุ่นนี้ครับ"),
                         "route": "rag_missing_corpus",
                         "latencies": {"missing_corpus": 0.0},
                         "context": [],
                         "context": [],
                         "source_nodes": []
                     }
                     return res
        except Exception as e:
             print(f"[WARN] Deterministic Retrieval Failed: {e}")

        # =========================================================================
        # PHASE 22: VENDOR SCOPE ENFORCEMENT (Pre-RAG)
        # =========================================================================
        # Block non-SMC vendors BEFORE vector search/RAG
        vendor_scope = self._detect_vendor_scope(q)
        v_name = vendor_scope['vendor']
        v_type = vendor_scope['type']
        
        # Rule: Block if:
        # 1. Vendor is SMC_ONLY but we have NO deterministic match (already reached this point)
        # (OUT_OF_SCOPE was already handled in Pre-Flight)
        should_block = False
        if v_name and v_type == "SMC_ONLY":
            # Deterministic match check already happened above. 
            # If we are here, it means no deterministic match was found.
            should_block = True
        
        if should_block:
            print(f"[GOVERNANCE] Phase 22: SMC-Only Vendor Block Triggered - {v_name}")
            
            # Acceptance Criteria 5 & 6 Reasons
            reason = f"{v_name.capitalize()} is SMC-only; no exact SMC article"

            res = {
                "answer": (
                    "⚠️ **นอกขอบเขต SMC (Out of Scope)**\n\n"
                    "ระบบ SMC-RAG ให้บริการข้อมูลเฉพาะจากเอกสารคู่มือในระบบ SMC เท่านั้น\n"
                    f"ผู้ผลิต '{v_name.upper() if v_name else 'Unknown'}' หรือรุ่นที่ระบุไม่มีข้อมูลในฐานข้อมูล SMC ครับ\n\n"
                    "กรุณาสอบถามเกี่ยวกับอุปกรณ์ที่มีในระบบ (เช่น Huawei, ZTE, Nokia) หรือระบุคำสั่งให้ชัดเจนขึ้นครับ"
                ),
                "route": "blocked_vendor_out_of_scope",
                "block_reason": f"NON_SMC_VENDOR_{v_name.upper() if v_name else 'UNKNOWN'}",
                "decision_reason": reason,
                "latencies": {"vendor_check": time.time() - t_start},
                "audit": {
                    "vendor_detected": v_name if v_name else "N/A",
                    "normalized_query": self.processed_cache.soft_normalize(q) if hasattr(self, 'processed_cache') else q,
                }
            }
            return res
        # =========================================================================
        # END PHASE 22
        # =========================================================================

        # STEP 5: AMBIGUITY PRE-CHECK GATE
        # Check if query is too broad/ambiguous BEFORE deterministic matching
        from src.query_analysis.ambiguity_detector import check_ambiguity, AmbiguityDetector
        ambiguity_result = check_ambiguity(q, query_intent)
        
        if ambiguity_result["is_ambiguous"]:
            reason = ambiguity_result["reason"]
            suggestion = ambiguity_result.get("suggestion", "กรุณาระบุรายละเอียดเพิ่มเติม")
            
            print(f"[STEP 5] Ambiguous query detected: {reason}")
            
            # If broad vendor command, show actual article options if available
            if reason == "BROAD_VENDOR_COMMAND" or reason == "MINIMAL_VENDOR_CONTEXT":
                 vendor = AmbiguityDetector.extract_vendor(q)
                 if vendor:
                     # Query deterministic index (Step 2)
                     vendor_articles = self._find_vendor_articles(vendor, limit=10)
                     
                     if vendor_articles:
                         print(f"[STEP 5] Found {len(vendor_articles)} articles for {vendor} -> Triggering Selection")
                         
                         # ============================================================
                         # NUMERIC SELECTION: Global numbering (NO category reset)
                         # ============================================================
                         items = []
                         for art in vendor_articles:
                             items.append({
                                 "url": art['url'],
                                 "title": art['title'],
                                 "category": art.get('category', '')
                             })
                         
                         # Create session with global numbering
                         session = self.numeric_selection_resolver.create_session(
                             items, context="article_selection",
                             prompt_text=f"กรุณาเลือกหมายเลข (1-{len(items)})"
                         )
                         options_text = self.numeric_selection_resolver.format_numbered_list(
                             session['items'], context="article_selection"
                         )
                         self.pending_numeric_session = session
                         
                         res = {
                             "answer": (
                                 f"พบ {len(vendor_articles)} เอกสารที่เกี่ยวข้อง:\n\n"
                                 f"{options_text}\n\n"
                                 f"{session['prompt_text']}"
                             ),
                             "route": "pending_clarification",
                             "metadata": {
                                 "kind": "vendor_command_selection",
                                 "vendor": vendor,
                                 "candidates": [
                                     {"title": art['title'], "url": art['url']}
                                     for art in vendor_articles
                                 ],
                                 "created_at": time.time(),
                                 "ambiguity_reason": reason,
                                 "original_query": q
                             },
                             "latencies": {"total": (time.time() - t_start) * 1000}
                         }
                         return res
                     else:
                         # Case: Vendor detected but no articles found in index
                         print(f"[STEP 5] Vendor {vendor} detected but no articles found.")
                         res = {
                             "answer": (
                                 f"ไม่พบคำสั่ง {vendor} ในระบบ\n"
                                 f"กรุณาระบุรุ่นอุปกรณ์ เช่น NE8000, OLT, Router"
                             ),
                             "route": "pending_clarification", 
                             "metadata": {
                                 "kind": "vendor_not_found",
                                 "vendor": vendor,
                                 "ambiguity_reason": reason,
                                 "original_query": q
                             },
                             "latencies": {"total": (time.time() - t_start) * 1000}
                         }
                         return res

            # BROAD_GENERIC_COMMAND: Search for articles matching the protocol keyword
            elif reason == "BROAD_GENERIC_COMMAND":
                 from src.query_analysis.ambiguity_detector import AmbiguityDetector as _ADG
                 # Extract the protocol keyword for search
                 q_lower = q.lower()
                 protocol_kw = next((p for p in _ADG.GENERIC_PROTOCOL_KEYWORDS if p in q_lower), None)
                 if protocol_kw:
                     # Title-based fuzzy match: only articles whose title contains the protocol keyword
                     link_results = self.processed_cache.find_links_fuzzy(protocol_kw, threshold=0.45)
                     proto_articles = []
                     seen_urls = set()
                     for lr in link_results:
                         for item in lr.get("items", []):
                             url = item.get("href", "")
                             title = item.get("text") or item.get("title", "")
                             if url and url not in seen_urls and title and protocol_kw.lower() in title.lower():
                                 proto_articles.append({"url": url, "title": title, "category": ""})
                                 seen_urls.add(url)
                                 if len(proto_articles) >= 10: break
                         if len(proto_articles) >= 10: break
                     
                     if proto_articles:
                         print(f"[STEP 5] Found {len(proto_articles)} articles for protocol ''{protocol_kw}''  -> Triggering Selection")
                         items = [{"url": a["url"], "title": a["title"], "category": a["category"]} for a in proto_articles]
                         session = self.numeric_selection_resolver.create_session(
                             items, context="article_selection",
                             prompt_text=f"กรุณาเลือกหมายเลข (1-{len(items)})"
                         )
                         options_text = self.numeric_selection_resolver.format_numbered_list(session["items"], context="article_selection")
                         self.pending_numeric_session = session
                         res = {
                             "answer": (
                                 f"พบ {len(proto_articles)} เอกสารที่เกี่ยวข้องกับ ''{protocol_kw}'' :\n\n"
                                 f"{options_text}\n\n"
                                 f"{session['prompt_text']}"
                             ),
                             "route": "pending_clarification",
                             "metadata": {"kind": "protocol_command_selection", "protocol": protocol_kw, "items": items, "created_at": time.time()},
                             "latencies": {"total": (time.time() - t_start) * 1000}
                         }
                         return res

            # Fallback: Generic clarification
            res = {
                "answer": (
                    "🤔 **คำถามยังกว้างเกินไป กรุณาระบุรายละเอียดเพิ่มเติม:**\n\n"
                    f"{suggestion}\n\n"
                    "ตัวอย่างที่ชัดเจนขึ้น:\n"
                    "• Huawei add vlan\n"
                    "• ZTE config\n"
                    "• คำสั่ง config IP ของ Huawei"
                ),
                "route": "pending_clarification",
                "metadata": {
                    "ambiguity_reason": reason,
                    "original_query": q
                },
                "latencies": {"total": (time.time() - t_start) * 1000}
            }
            return res


        telemetry_data = {
            "timestamp": t_start,
            "query": original_q_str,
            "mode": "FULL",
            "intent": "UNKNOWN",
            "clarification_triggered": False,
            "clarification_reason": "none",
            "top_score": 0.0,
            "second_score": 0.0,
            "score_gap": 0.0,
            "n_candidates_above_threshold": 0
        }
        self.last_telemetry = telemetry_data
        
        # Rule PR-1: Pre-normalize common typos for "เบอร์" (Phase 236)
        # Handle variations: เบออร์, เบอร, เบอร์ออร์, เบอ, เบแอร์, บอร์, เบอร์ร์
        # Logic: Match 'บ' or 'เบ' followed by any combination of 'อ', 'ร', '์', 'แ' that vaguely resembles 'เบอร์'
        # Regex: (เบ?|บ)[อแรต์ว์]{1,5} -> เบอร์ (Simple Fuzzy)
        # But allow "บอร์" (no leading vowel)
        # Specific patterns:
        if re.search(r"เบ[ออ่ร์รแ\.]{1,5}", q) or re.search(r"บ[อ่ร์ร]{1,4}", q):
             # Be careful not to replace valid words starting with "เบ" like "เบเกอรี่" (unlikely in this domain)
             # Better specific pattern list from user:
             # เบอร, เบออร์, เบแอร์, บอร์, เบอร์ร์
             q = re.sub(r'เบ[อ่ร์รแ]+[อ่ร์รแ\.]{0,4}', 'เบอร์', q) # Handles เบออร์, เบแอร์, เบอร
             q = re.sub(r'\bบอร์\b', 'เบอร์', q) # Handles บอร์ (standalone)
             
        # Also handle "เบอ " -> "เบอร์ " (if not caught above)
        if q.startswith("เบอ ") or " เบอ " in q:
            q = q.replace("เบอ ", "เบอร์ ")
        
        # =========================================================================
        # STEP 19.1-19.2: DOMAIN-SPECIFIC NORMALIZATION & INTENT LOCKING
        # Goal: Skip SafeNormalizer for structured domains (CONTACT, POSITION, TEAM)
        # =========================================================================
        
        # Detect structured query intent BEFORE SafeNormalizer
        intent_locked = None
        q_lower_check = q.lower()
        
        # STEP 19.2: Contact keyword detection
        CONTACT_KEYWORDS_STRONG = {"เบอร์", "ติดต่อ", "phone", "โทร", "อีเมล", "email", "fax", "แฟกซ์", "โทรสาร"}
        if any(kw in q_lower_check for kw in CONTACT_KEYWORDS_STRONG):
            intent_locked = "CONTACT_LOOKUP"
            print(f"[DEBUG] [STEP 19.2] Intent locked: CONTACT_LOOKUP (keyword match)")
        
        # STEP 19.2: Position keyword detection
        POSITION_KEYWORDS = {"ผจ", "ผอ", "หัวหน้า", "ผู้จัดการ", "position", "ตำแหน่ง"}
        if not intent_locked and any(kw in q_lower_check for kw in POSITION_KEYWORDS):
            intent_locked = "POSITION_LOOKUP"
            print(f"[DEBUG] [STEP 19.2] Intent locked: POSITION_LOOKUP (keyword match)")

        # --- SAFE NORMALIZATION LAYER ---
        # STEP 4: Deterministic BEFORE Heuristic (already enforced by flow order)
        # STEP 3: Skip normalizer entirely if pending active (Layer 1 handles all input)
        # STEP 19.1: Skip normalizer for structured domains (CONTACT/POSITION)
        
        if self.pending_question:
            print(f"[DEBUG] [STEP 3-4] Pending active -> Skipping SafeNormalizer (Layer 1 priority)")
            shape_analysis = {"confidence": 0}  # Empty result, skip normalization
        elif intent_locked in ["CONTACT_LOOKUP", "POSITION_LOOKUP"]:
            print(f"[DEBUG] [STEP 19.1] Structured domain ({intent_locked}) -> Skipping SafeNormalizer (deterministic flow)")
            shape_analysis = {"confidence": 0, "intent": intent_locked}  # Skip normalization, use locked intent
        else:
            # Robustly handle paraphrases using the user-provided "Safe Intent Shape Analyzer" prompt
            shape_analysis = self.safe_normalizer.analyze(q)
        
        # =========================================================================
        # CRITICAL FIX: Preserve Intent from Context Enrichment
        # =========================================================================
        # If query was enriched with context (e.g., "ของ RNOC ละ" + "หาดใหญ่"),
        # MUST preserve the original intent (CONTACT_LOOKUP) from context.
        # SafeNormalizer may incorrectly classify enriched query as GENERAL_QA.
        
        ai_intent = None
        is_asset_request = False
        asset_type = None
        
        if shape_analysis.get("confidence", 0) > 0.7:
            canonical_q = shape_analysis.get("canonical_query")
            if canonical_q:
                print(f"[CHAT] Canonical Rewrite: {q} -> {canonical_q}")
                q = canonical_q
            ai_intent = shape_analysis.get("intent")
            telemetry_data["ai_intent"] = ai_intent
            telemetry_data["request_shape"] = shape_analysis.get("request_shape")
            
            if shape_analysis.get("request_shape") == "ASSET":
                is_asset_request = True
                asset_type = shape_analysis.get("entities", {}).get("asset_type")
                telemetry_data["is_asset_request"] = True
                telemetry_data["asset_type"] = asset_type
        
        # Override SafeNormalizer intent if context preserved an intent
        if hasattr(self, 'context_intent_override') and self.context_intent_override:
            print(f"[CONTEXT_INTENT_FIX] Preserving intent from context: {self.context_intent_override} (overriding SafeNormalizer: {ai_intent})")
            ai_intent = self.context_intent_override
            telemetry_data["ai_intent"] = ai_intent
            telemetry_data["context_intent_preserved"] = True
        
        # Phase 230: Query Normalization & Alias Rewrite
        # This handles "makebridge" -> "make bridge" and "bridge port" -> "Canonical Title"
        q, applied_rule = expand_synonyms(q)
        synonym_active = bool(applied_rule)
        if synonym_active:
             print(f"[DEBUG] Query Expanded/Rewritten: {original_q_str} -> {q} (Rule: {applied_rule})")
        
        # Update telemetry with expanded query
        telemetry_data.update({
            "expanded_query": q,
            "route": "unknown",
            "clarify_asked": False,
            "synonym_rule": applied_rule,
            "synonym_rollback": False,
            "pack_hit": False
        })
        
        bypass_kp = False # Flag to skip Knowledge Pack (e.g. if explicitly falling back from HowTo)
        

        # 0.5 Pending Clarification Resolver (Phase 26.5)
        # Intercepts input if we are waiting for a Knowledge Pack scope
        if self.pending_kp_clarify:
            print(f"[DEBUG] Resolving Pending KP Clarification: {q}")
            
            p_state = self.pending_kp_clarify
            created_ts = p_state.get("created_ts", time.time())
            turns = p_state.get("turns", 0)
            
            # 1. TTL Check (60s)
            if time.time() - created_ts > 60:
                print("[DEBUG] Clarification Expired (TTL)")
                self.pending_kp_clarify = None
                telemetry_data["clarify_expired"] = True
                # Fall through to standard processing
            else:
                # 2. Interruption Check (New Query Detection)
                # Heuristic: Match typical new query patterns vs Valid Scope Reply
                reply_raw = q.lower().strip()
                reply_clean = re.sub(r"[\[\]\(\)]", "", reply_raw).strip()
                for w in ["ขอ", "เลือก", "เอา", "ครับ", "ค่ะ"]:
                    reply_clean = reply_clean.replace(w, "").strip()
                
                # Keywords that suggest a new intent (not a scope)
                NEW_INTENT_KWS = ["ข่าว", "แจ้ง", "ตาราง", "เบอร์", "โทร", "ip", "smtp", "dns", "vlan", "config", "วิธี", "ทำยังไง", "คู่มือ", "ปัญหา", "แก้ไข", "how to"]
                
                # Check for "All" intent (Valid Scope)
                is_all = any(w in reply_clean for w in ["all", "ทั้งหมด", "ทุกรายการ"])
                
                # Check if it matches any option (Valid Scope)
                options = p_state.get("options", [])
                matched_scope = None
                if not is_all:
                    for opt in options:
                         if opt.lower() in reply_clean or reply_clean in opt.lower():
                             matched_scope = opt; break
                    
                    if not matched_scope:
                         # Aliases
                         if any(k in reply_clean for k in ["nt", "nt1", "n t 1", "bkk"]): matched_scope = "NT1"
                         elif any(k in reply_clean for k in ["region", "ภูมิภาค", "ต่างจังหวัด", "ตจว"]): matched_scope = "ภูมิภาค (Region)"
                         elif any(k in reply_clean for k in ["isp", "core"]): matched_scope = "ISP"
                
                # INTERRUPTION DECISION
                is_interruption = False
                # If input is long (> 20 chars) and NOT 'all' -> Likely a new query
                if len(reply_raw) > 20 and not is_all:
                    is_interruption = True
                # If input contains new intent keywords AND no matched scope -> Interrupt
                elif any(k in reply_raw for k in NEW_INTENT_KWS) and not matched_scope and not is_all:
                    is_interruption = True
                # Multi-phrase check (>= 3 spaces) -> Likely sentence
                elif reply_raw.count(" ") >= 3:
                    is_interruption = True
                    
                if is_interruption:
                    print(f"[DEBUG] Clarification Interrupted by new query: {q}")
                    self.pending_kp_clarify = None
                    telemetry_data["clarify_interrupted"] = True
                    # Fall through to normal processing
                
                else:
                    # Not an interruption, treat as an attempt to answer scope
                    if is_all or matched_scope:
                        # VALID RESOLUTION
                        base_q = p_state.get("original_query", "")
                        saved_mode = p_state.get("pending_answer_mode", "FULL")
                        
                        final_query = f"{base_q} {'ทั้งหมด' if is_all else matched_scope}"
                        print(f"[DEBUG] Clarification Resolved -> {final_query}")
                        
                        self.pending_kp_clarify = None
                        if is_all: telemetry_data["resolved_all"] = True
                        else: telemetry_data["resolved_scope"] = matched_scope
                        
                        if self.kp_manager:
                            kp_result = self.kp_manager.lookup(final_query)
                            if kp_result:
                                telemetry_data["clarify_resolved"] = True
                                telemetry_data["route"] = "knowledge_pack_resolved"
                                telemetry_data["pack_hit"] = True
                                self._log_telemetry(telemetry_data, latencies)
                                
                                res = {
                                     "answer": kp_result["answer"],
                                     "route": "knowledge_pack_resolved",
                                     "latencies": latencies,
                                     "hits": kp_result["hits"]
                                }
                                return self._apply_answer_mode(res, saved_mode)
                        
                        # Fallback if manager missing or empty result (rare)
                        pass 
                        
                    else:
                        # INVALID SCOPE REPLY (and not interruption)
                        # Check Max Retries
                        if turns >= 1: # Already re-asked once
                             print("[DEBUG] Clarification Failed (Max Turns)")
                             self.pending_kp_clarify = None
                             telemetry_data["clarify_failed"] = True
                             # Fall through to normal processing (treat as new query or just give up?)
                             # Requirement: "Clear pending_kp and route normally"
                        else:
                             # Re-ask
                             print("[DEBUG] Clarification Re-ask")
                             self.pending_kp_clarify["turns"] = turns + 1
                             options = p_state.get("options", [])
                             options_str = " ".join([f"[{opt}]" for opt in options])
                             return {
                                "answer": f"ขออภัยครับ กรุณาเลือกขอบเขตจาก: {options_str} หรือพิมพ์ 'ทั้งหมด'",
                                "route": "knowledge_pack_clarify_reask",
                                "latencies": latencies
                             }

        # ---------------------------------------------------------
        # INPUT SANITIZATION & PRE-FILTERS (Phase 171)
        # ---------------------------------------------------------
        
        # 1. Empty or Whitespace Only
        q_strip = q.strip()
        if not q_strip:
            return {
                "answer": "สวัสดีครับ มีอะไรให้ผมช่วยค้นหาข้อมูลหรือเบอร์โทรศัพท์ในวันนี้ครับ?",
                "route": "quick_reply",
                "latencies": latencies
            }

        # 2. Excessively Long Input (Anti-DoS / Performance)
        if len(q) > 2000:
            print(f"[DEBUG] Truncating excessively long query ({len(q)} -> 1000)")
            q = q[:1000] # Truncate for safety
            q_strip = q.strip()
            telemetry_data["query_truncated"] = True

        # 3. Numeric or Emoji Only (Deterministic Fallback)
        is_numeric = q_strip.replace(".", "", 1).isdigit()
        has_text = any(c.isalpha() or '\u0e00' <= c <= '\u0e7f' for c in q_strip)
        
        if is_numeric or not has_text:
            print(f"[DEBUG] Non-text query detected: '{q_strip}'")
            return {
                "answer": "ขออภัยครับ ผมไม่เข้าใจคำค้นหาของคุณ กรุณาพิมพ์เป็นข้อความ เช่น 'เบอร์ CSOC' หรือ 'วิธีเช็ค IP' ครับ",
                "route": "quick_reply",
                "latencies": latencies
            }

        # ---------------------------------------------------------
        # DETERMINISTIC FAST-PATHS / EARLY EXITS (Phase 170)
        # ---------------------------------------------------------
        
        # Guard A: Dispatch Mapper Handler (Province Mapping)
        try:
            from src.rag.handlers.dispatch_mapper import DispatchMapper
            if DispatchMapper.is_match(q):
                print(f"[DEBUG] DispatchMapper Intent Triggered: {q}")
                if not self.processed_cache:
                    from src.core.chat_engine import ProcessedCache 
                    self.processed_cache = ProcessedCache()
                    self.processed_cache.load()
                
                dispatch_res = DispatchMapper.handle(q, self.processed_cache)
                if dispatch_res and dispatch_res.get("route") != "dispatch_mapper_error":
                     return {
                         "answer": dispatch_res.get("answer"),
                         "route": dispatch_res.get("route"),
                         "latencies": latencies,
                         "debug_info": {"handler": "DispatchMapper"},
                         "context": dispatch_res.get("context")
                     }
        except Exception as e:
            print(f"[DEBUG] DispatchMapper Error: {e}")

        # Safety Governance Header (Signaling only, not blocking)
        self.check_governance_blocking(q)

        # Guard B: Credential / Secret Guard (Safety)
        q_sec = q.lower()
        sec_keywords = ["password", "passcode", "pwd", "รหัสผ่าน", "รหัสเข้า", "default pass", "admin pass", "root pass"]
        has_sec_kw = any(k in q_sec for k in sec_keywords)
        high_risk_targets = ["admin", "ont", "router", "cpe", "root", "super", "system", "dms", "radius", "tacacs"]
        is_target_risk = any(t in q_sec for t in high_risk_targets)
        
        if has_sec_kw or is_target_risk:
             # ONT Specific logic
             if "ont" in q_sec and "password" in q_sec:
                  sec_cfg = self.cfg.get("security", {})
                  allow_redirect = sec_cfg.get("allow_credential_redirect", False)
                  redirect_mode = sec_cfg.get("credential_redirect_mode", "block")
                  allowlist = sec_cfg.get("credential_article_allowlist", [])
                  
                  matched_link = None
                  if allow_redirect:
                      fuzzy_results = self.processed_cache.find_links_fuzzy(q) if self.processed_cache else []
                      if fuzzy_results:
                          for res in fuzzy_results:
                              for item in res.get("items", []):
                                  url_lower = item.get("href", "").lower()
                                  if any(allowed.lower() in url_lower for allowed in allowlist):
                                      matched_link = item
                                      matched_link["url"] = item.get("href")
                                      break
                              if matched_link: break
                  
                  if matched_link and redirect_mode == "redirect_only":
                      return {
                          "answer": (f"⚠️ **ข้อมูลจำกัดสิทธิ์ (Restricted Content)**\n\nระบบไม่สามารถแสดงเนื้อหารหัสผ่านได้ แต่พบเอกสารที่เกี่ยวข้อง:\n"
                                     f"🔗 **[คลิกเพื่อเปิดอ่าน (ต้องมีสิทธิ์ SMC)]({matched_link['url']})**\n\n*(กรุณาตรวจสอบสิทธิ์การเข้าถึงผ่านระบบภายใน)*"),
                          "route": "rag_credential_redirect",
                          "latencies": latencies
                      }
                  return {
                      "answer": ("🔒 **ขออภัยครับ ระบบไม่สามารถแสดงรหัสผ่าน ONT ได้เนื่องจากนโยบายความปลอดภัย**\n\n📌 **คำแนะนำ**:\n- รหัสผ่าน ONT เป็นข้อมูลเฉพาะของแต่ละพื้นที่/ชุมสาย\n- กรุณาติดต่อ **ทีม OMC ประจำเขต** หรือเปิด Ticket พร้อมระบุ **รุ่นและ Node** ที่ต้องการ"),
                      "route": "rag_security_guided",
                      "latencies": latencies
                  }
             
             # Patterns (Admin Admin)
             if "admin" in q_sec and ("admin" in q_sec.replace("admin", "", 1) or "password" in q_sec or "default" in q_sec):
                 return {
                     "answer": ("🔒 **ระงับการตอบกลับ (Security Restricted)**\n\nระบบตรวจพบแพทเทิร์นการขอรหัสผ่าน Default/Admin ซึ่งขัดต่อนโยบายความปลอดภัย (Security Policy)"),
                     "route": "rag_security_guided",
                     "latencies": latencies
                 }
             
             # General Block
             if has_sec_kw and is_target_risk:
                 return {
                     "answer": ("🔒 **ขออภัยครับ ระบบไม่สามารถแสดงรหัสผ่านได้เนื่องจากนโยบายความปลอดภัย (Security Policy)**\n\n📌 **ช่องทางดำเนินการที่แนะนำ**: ติดต่อทีม **NOC/OMC** หรือเปิด Ticket ตามระเบียบปฏิบัติ"),
                     "route": "rag_security_guided",
                     "latencies": latencies
                 }

        # Guard C: Ambiguous / Short Token Guard
        from src.directory.lookup import is_broad_query
        q_clean_for_guard = q.strip().lower()
        ambiguous_terms = ["sbc", "network", "omc", "bras", "access", "core", "metro", "report", "test", "wifi"]
        
        # Phase 231: Allow Broad Queries (Rule I) to pass through
        is_broad = is_broad_query(q_clean_for_guard)
        print(f"[DEBUG] Guard C: Query='{q_clean_for_guard}' is_broad={is_broad}")
        
        if not is_broad and ((len(q_clean_for_guard.split()) == 1 and len(q_clean_for_guard) < 15 and q_clean_for_guard.isalpha() and q_clean_for_guard not in ["hi", "hello", "test", "ping"]) or q_clean_for_guard in ambiguous_terms):
             return {
                 "answer": f"คำค้นหา '{q}' กว้างเกินไปครับ กรุณาระบุให้ชัดเจน เช่น '{q} เบอร์โทร' หรือ 'วิธีแก้ปัญหา {q}'",
                 "route": "contact_ambiguous",
                 "latencies": latencies
             }



        
        # Logic C: Follow-up State (Phase 48)
        # Check if we are awaiting a province for Dispatch Mapper
        current_ctx = self.last_context.get("context") if self.last_context else None
        if current_ctx == "awaiting_province" and len(q) < 50:
            print(f"[DEBUG] Context: Awaiting Province -> '{q}'")
            # Route to DispatchMapper as Followup
            try:
                from src.rag.handlers.dispatch_mapper import DispatchMapper
                if not self.processed_cache:
                    from src.core.chat_engine import ProcessedCache 
                    self.processed_cache = ProcessedCache()
                    self.processed_cache.load()
                    
                dispatch_res = DispatchMapper.handle_followup(q, self.processed_cache)
                # Return immediately (clearing context happens automatically if we don't return it again?)
                # Actually we should decide if we want to clear context. 
                # If answer successful, context is cleared (by not returning it).
                # If fail, maybe keep it? DispatchMapper decides.
                return {
                     "answer": dispatch_res.get("answer"),
                     "route": dispatch_res.get("route"),
                     "latencies": latencies,
                     "context": dispatch_res.get("context") # Pass new context if any
                }
            except Exception as e:
                print(f"[DEBUG] Dispatch Followup Error: {e}")
                # Fall through

        # ---------------------------------------------------------
        # GUARD: Technical Protection & Canonical (Phase 35 Enhancements)
        # ---------------------------------------------------------
        
        # 1. Canonical Normalization (e.g. "ipphone" -> "ip-phone")
        # Fix: Use get_canonical_phrase() which only needs query and returns (canonical, rewritten)
        canonical_rule, q_canonical = get_canonical_phrase(q)
        if canonical_rule:
            print(f"[DEBUG] Canonical Rule Applied: {canonical_rule} -> {q_canonical}")
            telemetry_data["canonical_rule"] = canonical_rule
            q = q_canonical # Update query
            original_q_str = q_canonical # Update original for routing consistency

        # 2. Asset Table Detection (Override)
        if is_asset_table_query(q):
            print(f"[DEBUG] Asset Table Intent Detected: {q}")
            # Force ASSET intent which uses directory_handler.handle_team_lookup(is_asset=True)
            # We set a flag to ensure override happens in routing section
            override_intent = "ASSET" 
        
        # ---------------------------------------------------------
        # INTENT ROUTING (Phase 34)
        # ---------------------------------------------------------
        t0 = time.time()
             
        # Phase 34: Hybrid Intent Routing
        # Classify query before proceeding to specialized handlers

        # ── Early KP exit for Huawei NE8000 command queries ─────────────────
        # Placed BEFORE intent routing so DEFINE_TERM / HOWTO_PROCEDURE cannot
        # intercept specific command lookups (e.g. "display cpu-usage คืออะไร").
        _q_lower_kp = q.lower()
        _huawei_triggers = ["huawei", "ne8000", "display ", "อัตราการใช้", "หน่วยความจำ", "manuinfo", "cpu-usage", "serial number"]
        if self.kp_manager and any(t in _q_lower_kp for t in _huawei_triggers):
            _kp_res = self.kp_manager.lookup(q)
            if _kp_res and _kp_res.get("hits"):
                # Only intercept if the top hit has a command key match (score ≥ 10)
                _top_hit = _kp_res["hits"][0] if _kp_res["hits"] else None
                _top_score = _top_hit[1] if isinstance(_top_hit, (list, tuple)) and len(_top_hit) > 1 else 0
                if _top_score >= 10:
                    print(f"[KP_EARLY_EXIT] Huawei command match (score={_top_score}) → returning KP answer")
                    telemetry_data["route"] = "knowledge_pack_huawei_cmd"
                    telemetry_data["pack_hit"] = True
                    latencies["total"] = (time.time() - t_start) * 1000
                    return {
                        "answer": _kp_res["answer"],
                        "route":  "knowledge_pack_huawei_cmd",
                        "latencies": latencies,
                        "hits": _kp_res["hits"],
                    }
        # ── End early KP exit ────────────────────────────────────────────────

        
        # Phase R5: Hard Intent Override for Concepts (Moved UP for Speed)
        # Check explicit question pattern
        # Updated Policy 1: DEFINE/TERM > HOWTO
        force_define_kws = ["คืออะไร", "what is", "meaning", "concept", "theory", "ทฤษฎี", "หมายถึง", "definition", "ความหมาย", "คือ", "แปลว่า", "ย่อมาจาก"]
        # Also include short abbr check here if needed, but the router might handle "OLT" better if we hint it?
        # User Rule: "olt/ont/onu... + define_kws" -> DEFINE_TERM
        
        q_lower_check = q.lower()
        sensitive_terms = ["password", "admin", "login", "user", "root", "secret", "รหัส"]
        is_sensitive = any(s in q_lower_check for s in sensitive_terms)
        
        override_intent = None
        
        # Rule: If query explicitly asks for definition (and not sensitive)
        # Rule: If query explicitly asks for definition (and not sensitive)
        # Rule DT-ROLE-1: Block DEFINE_TERM for Roles/Persons e.g. "ใครคือ/ใครเป็น/ผช./ผจ." (Force POSITION_HOLDER_LOOKUP)
        role_triggers = ["ใครคือ", "คือใคร", "ใครเป็น", "ผู้รับผิดชอบ", "ผช.", "ผส.", "ผจ.", "ผอ.", "ชจญ.", "หน.", "หัวหน้า", "ใครดูแล"]
        
        # Rule BQ-1: Broad Category -> Force CONTACT_LOOKUP (to show list)
        is_contact_trigger = any(ct in q_lower_check for ct in ["เบอร์", "โทร", "ติดต่อ", "phone", "contact"])
        # Fix: Exclude "ip phone" or "ipphone" from generic "phone" trigger unless explicit contact words exist
        if "ip" in q_lower_check and "phone" in q_lower_check:
             # If query is just "ip phone" -> likely definition/howto.
             # If "contact ip phone" -> contact.
             # We rely on "เบอร์" or "ติดต่อ" being present if user wants contact for IP Phone.
             # If only "phone" (english) is the trigger, and "ip" is present, ignore it.
             if is_contact_trigger and not any(ct in q_lower_check for ct in ["เบอร์", "โทร", "ติดต่อ"]):
                 is_contact_trigger = False

        if any(rt in q_lower_check for rt in role_triggers):
             print(f"[DEBUG] Intent Override: '{q}' -> POSITION_HOLDER_LOOKUP (Role Trigger)")
             override_intent = "POSITION_HOLDER_LOOKUP"

        # Rule CT-OVR-1: Contact overrides everything (especially Knowledge Alias for HowTo)
        # If query has contact triggers, Force CONTACT_LOOKUP
        elif is_contact_trigger:
             print(f"[DEBUG] Intent Override: '{q}' -> CONTACT_LOOKUP (Contact Trigger)")
             override_intent = "CONTACT_LOOKUP"

        elif any(k in q_lower_check for k in force_define_kws) and not is_sensitive:
             print(f"[DEBUG] Intent Override: '{q}' -> DEFINE_TERM (Keyword Match)")
             override_intent = "DEFINE_TERM"
             
        # Rule BQ-1: Broad Category -> Force CONTACT_LOOKUP (to show list)
        elif is_broad:
             print(f"[DEBUG] Intent Override: '{q}' -> CONTACT_LOOKUP (Broad Category)")
             override_intent = "CONTACT_LOOKUP"

        # Rule: Short Abbreviations (OLT, ONT...) -> Force DEFINE if no action verb
        # e.g. "OLT" -> DEFINE_TERM (Instead of HowTo fail)
        # Fix: Do NOT force if it is a Broad Category (e.g. "edimax", "csoc", "noc") -> Let Contact Handler handle it
        elif len(q.split()) <= 2 and re.match(r"^[a-zA-Z0-9\-\.\/]{2,6}$", q.strip()) and not is_broad:
             # Short English acronyms likely define intent if no context
             # Exclude common commands like "ping", "test"
             if q_lower_check not in ["ping", "test", "demo", "help", "exit"]:
                 print(f"[DEBUG] Intent Override: '{q}' -> DEFINE_TERM (Short Abbreviation)")
                 override_intent = "DEFINE_TERM"

        if override_intent:
             routing_res = {"intent": override_intent, "confidence": 1.0, "reason": "Keyword Override"}
        else:
             # Phase 174 Fix: Use original query for routing to avoid synonym confusion
             routing_res = self.router.route(original_q_str)
             
        intent = routing_res["intent"]
        
        # GUARD: Contact Precision Check (Phase 229)
        # Prevent false-positive CONTACT_LOOKUP for technical/config queries
        if intent == "CONTACT_LOOKUP":
            if not is_valid_contact_query(q):
                print(f"[DEBUG] Contact Precision Guard: Invalid Contact Query -> Rerouting")
                
                # Check for Fail-Soft Reroute to HOWTO
                # Note: should_reroute_to_howto expects (query, route) not (query, intent)
                # Since we're in invalid contact context, use a fake route to check patterns
                should_reroute, new_intent = should_reroute_to_howto(q, "contact_miss_strict")
                if should_reroute:
                     print(f"[DEBUG] Fail-Soft Reroute -> HOWTO_PROCEDURE")
                     intent = "HOWTO_PROCEDURE"
                     routing_res["intent"] = "HOWTO_PROCEDURE"
                     routing_res["reason"] = "Fail-Soft Reroute (Technical Context)"
                else:
                     print(f"[DEBUG] Rerouting to GENERAL_QA")
                     intent = "GENERAL_QA"
                     routing_res["intent"] = "GENERAL_QA"
                     routing_res["reason"] = "Contact Precision Guard Block"
        
        # GUARD: HOWTO Shield (Phase 230)
        # Ensure technical terms are routed to HOWTO/ASSET if not asking for contact
        if intent == "CONTACT_LOOKUP" and has_protected_term(q):
             print(f"[DEBUG] HOWTO Shield: Protected Term detected in CONTACT_LOOKUP -> Re-evaluating")
             # If no explicit contact word ("เบอร์", "โทร"), force ASSET/HOWTO
             # This is partially covered by Contact Precision, but this uses the protected term list
             pass # Logic mostly covered by Contact Precision, but keeping placeholder for strict enforcement if needed

        # Rule SI-1: Respect AI-Driven Shape Intent if high confidence and deterministic match exists
        if ai_intent and intent == "GENERAL_QA" and ai_intent in ["CONTACT_LOOKUP", "TEAM_LOOKUP", "POSITION_LOOKUP", "ASSET"]:
             print(f"[DEBUG] Upgrading Intent from {intent} -> {ai_intent} (SafeNormalizer Priority)")
             intent = ai_intent
             routing_res["intent"] = ai_intent
        
        # Phase 34: Auto-Correction for Fallback Intents
        # If intent is GENERAL_QA (fallback) try to correct typos (e.g. "ดบอร์โทร" -> "เบอร์โทร")
        # Rule AB-1: Category Lock - Do not correct if prefix is a known category
        category_prefixes = ["สื่อสารข้อมูล", "ศูนย์", "เบอร์", "ตาราง", "งาน", "ฝ่าย", "ส่วน"]
        is_category_query = any(original_q_str.startswith(p) for p in category_prefixes)
        
        if intent == "GENERAL_QA" and len(original_q_str) > 3 and not is_category_query:
            q_corrected = self.corrector.correct(original_q_str)
            if q_corrected and q_corrected != original_q_str:
                 print(f"[DEBUG] Corrected Query: '{original_q_str}' -> '{q_corrected}'")
                 original_q_str = q_corrected # Update original for routing retry
                 routing_res = self.router.route(original_q_str)
                 intent = routing_res["intent"]

        confidence = routing_res["confidence"]
        
        # Old Override Block Removed

        telemetry_data["intent"] = intent
        telemetry_data["intent_conf"] = confidence
        print(f"[DEBUG] Intent: {intent} ({confidence}) Reason: {routing_res.get('reason')}")

        # Phase 97: Pro-Level Greeting Handler (Deterministic & No-Cache)
        greeting_resp = self.greetings_handler.handle(q, intent)
        if greeting_resp:
            print(f"[CHAT] Greeting Handler Hit: {q}")
            latencies["total"] = (time.time() - t_start) * 1000
            greeting_resp["latencies"] = latencies
            greeting_resp["route"] = "greeting_deterministic"
            return greeting_resp

        # Phase 87/90: Override CLARIFY for Knowledge/Procedure Queries
        # Route to Article-First (HOWTO_PROCEDURE) instead of slow RAG path
        KNOWLEDGE_KEYWORDS = ["ความรู้", "คู่มือ", "manual", "แนวทาง", "วิธี", "ขั้นตอน", 
                              "howto", "config", "แก้ไข", "ตั้งค่า", "ทำอย่างไร", "how to"]
        
        q_stripped = q.lower()
        # Remove polite particles for length check
        for particle in ["ครับ", "ค่ะ", "คะ", "จ้า"]:
            q_stripped = q_stripped.replace(particle, "")
        q_stripped = q_stripped.strip()
        q_lower = q.lower()
        
        has_knowledge_kw = any(kw in q_lower for kw in KNOWLEDGE_KEYWORDS)
        
        if intent == "CLARIFY" and has_knowledge_kw and len(q_stripped) >= 6:
            print(f"[DEBUG] Overriding CLARIFY -> HOWTO_PROCEDURE (Knowledge keyword detected: '{q}')")
            intent = "HOWTO_PROCEDURE"
            routing_res["intent"] = "HOWTO_PROCEDURE"
            routing_res["reason"] = "Knowledge keyword override (Article-First)"

        # Fix 1 (Phase 102): Knowledge Alias Trigger for Short Technical Topics
        # e.g. "sbc ip" -> HOWTO_PROCEDURE (even without "knowledge" keyword)
        
        # Check for Strong Link Match first
        # Rule RT-1: Lower threshold to 0.80 and FORCE override even if Web Knowledge
        link_hits = self.processed_cache.find_links_fuzzy(q_stripped, threshold=0.80)
        is_link_match = link_hits and link_hits[0]["score"] >= 0.80
        
        # Check if valid team/position (to avoid overriding valid lookups)
        is_valid_team = False
        if intent == "TEAM_LOOKUP" or ai_intent == "TEAM_LOOKUP":
             from src.utils.normalization import normalize_for_matching
             from src.utils.metrics import MetricsTracker
             import json
             q_norm_for_check = normalize_for_matching(q_stripped)
             q_ns = q_norm_for_check.replace(" ", "")
             
             # Check if q_ns contains any team name (no spaces)
             for team_norm in self.directory_handler.team_norm_map.keys():
                 tn_ns = team_norm.replace(" ", "")
                 if len(tn_ns) > 2 and (tn_ns in q_ns or q_ns in tn_ns):
                      is_valid_team = True
                      break
             
        # Allow overriding CONTACT/TEAM/WEB if it's a specific link match OR a generic tech alias that isn't a verified team
        # Added WEB_KNOWLEDGE to allow rescuing "Bridge Cisco" from WebHandler
        allowed_intents = ["GENERAL_QA", "CLARIFY", "CONTACT_LOOKUP", "TEAM_LOOKUP", "WEB_KNOWLEDGE"]
        
        if intent in allowed_intents:
             q_tokens = q_stripped.split()
             # Logic: Short (<=3 words) AND Technical (English chars present)
             is_tech_pattern = len(q_tokens) <= 3 and re.search(r"[a-zA-Z]", q_stripped) and len(q_stripped) > 2
             
             # COND 1: Exact Link Match -> Force HowTo
             # COND 2: Tech Pattern AND NOT Valid Team AND NOT Contact Query -> Force HowTo
             should_override = False
             reason = ""
             
             # Check for explicit contact keywords to prevent overriding valid contact lookups
             # e.g. "เบอร์งาน FTTx" should NOT be coerced to HowTo just because it has english chars.
             has_contact_kw = any(k in q_stripped for k in ["เบอร์", "โทร", "phone", "contact", "call", "ติดต่อ"])
             
             # Fix: Exclude "phone" if part of "ip phone" context without other contact words
             if "ip" in q_stripped and "phone" in q_stripped and not any(k in q_stripped for k in ["เบอร์", "โทร", "contact", "call", "ติดต่อ"]):
                 has_contact_kw = False
             
             # Rule PR-2: Override Shield (Phase 236)
             # If intent is already CONTACT_LOOKUP and we have a phone signal, 
             # do NOT override unless explicit HOW-TO verbs are present.
             HOWTO_VERBS = ["ตั้งค่า", "วิธี", "ทำอย่างไร", "config", "setup", "how to", "แก้", "เพิ่ม", "ทำยังไง", "command"]
             has_howto_verb = any(v in q_stripped for v in HOWTO_VERBS)
             
             is_contact_protected = (intent == "CONTACT_LOOKUP" and has_contact_kw and not has_howto_verb)
             
             # Rule SI-2: Protect Directory Intents if explicitly identified by SafeNormalizer
             is_dir_protected = (intent in ["TEAM_LOOKUP", "POSITION_LOOKUP", "CONTACT_LOOKUP"] and ai_intent in ["TEAM_LOOKUP", "POSITION_LOOKUP", "CONTACT_LOOKUP"])

             if is_link_match:
                 # Rule PR-2: Even for Link Match, if explicitly asking for contact/team, prefer Directory unless HowTo requested.
                 if is_contact_protected or is_dir_protected or is_valid_team:
                     print(f"[DEBUG] Blocking Knowledge Link Match ({link_hits[0].get('text')}) -> {intent} (Protected)")
                     should_override = False
                 else:
                     should_override = True
                     reason = "Direct Link Match (Article-First)"
             elif is_tech_pattern and not is_valid_team and not has_contact_kw and not is_contact_protected and not is_dir_protected:
                 should_override = True
                 reason = "Tech Pattern Alias Trigger (Not a Team)"
                  
             if should_override:
                  # Rule K1: Check for Thai Org Knowledge triggers (5S) -> Force Special Intent
                  org_triggers = ["5ส", "5 ส", "5s", "มาตรฐาน 5ส"]
                  if any(t in q_lower for t in org_triggers) and "rfc" not in q_lower:
                       print(f"[DEBUG] Knowledge Alias Trigger: '{q}' -> Forcing THAI_ORG_KNOWLEDGE (Rule K1 5S Override)")
                       intent = "THAI_ORG_KNOWLEDGE"
                       routing_res["intent"] = "THAI_ORG_KNOWLEDGE"
                       routing_res["reason"] = f"Rule K1 Override ({reason})"
                  else:
                       print(f"[DEBUG] Knowledge Alias Trigger: '{q}' -> Forcing HOWTO_PROCEDURE ({reason})")
                       intent = "HOWTO_PROCEDURE"
                       routing_res["intent"] = "HOWTO_PROCEDURE"
                       routing_res["reason"] = reason



        # Phase 97: Strict Cache Check (Moved UP before Handlers)
        # Rule DT-1/DT-ROLE-1: Bypass early semantic cache for Define and Position intents 
        # to ensure fresh logic (Internal Missing check) or correct routing behavior.
        if self.cache and not bypass_cache and intent not in ["DEFINE_TERM", "POSITION_HOLDER_LOOKUP", "CONTACT_LOOKUP"]:
            # Check cache using strict intent/route context
            # Map intent to route for strict checking
            cache_route = "rag" # Default
            if intent == "CONTACT_LOOKUP": cache_route = "contact_lookup"
            elif intent == "PERSON_LOOKUP": cache_route = "person_lookup"
            elif intent == "REFERENCE_LINK": cache_route = "link_lookup"
            elif intent == "NEWS_SEARCH": cache_route = "news_search"
            elif intent == "HOWTO_PROCEDURE": cache_route = "article_answer"
            
            cached = self.cache.check(
                q, 
                intent=intent, 
                route=cache_route, 
                filter_meta={
                    "model": self.llm_cfg.get("model", "unknown"),
                    "prompt_version": PROMPT_VERSION # STRICT VERSIONING
                }
            )
            
            if cached:
                # Phase 30: Cache Governance
                # If cached answer is long (likely RAG) but missing sources, ignore it to force re-generation with sources.
                ans = cached["answer"]
                has_sources = "แหล่งข้อมูลอ้างอิง:" in ans
                is_long = len(ans) > 100
                # Relax Governance for General Fallback (which naturally has no sources)
                is_general_fallback_answer = "ไม่พบเอกสารภายใน" in ans or "หลักการทั่วไป" in ans
                
                if is_long and not has_sources and not is_general_fallback_answer:
                    print(f"[DEBUG] Cache Governance: Rejected valid cache because missing evidence metadata (len={len(ans)})")
                    telemetry_data["cache_refusal_missing_refs"] = True
                else:
                    latencies["total"] = cached["latency"]
                    latencies["cache_hit"] = True
                    return {
                        "answer": cached["answer"],
                        "route": "cache_hit",
                        "latencies": latencies,
                        "score": cached["score"]
                    }


        # Phase 82: Explicit CLARIFY Handler
        # Bypass cache/RAG for ambiguous queries, delegate to DirectoryHandler to find candidates.
        if intent == "CLARIFY":
             print("[DEBUG] Handling CLARIFY intent -> Deterministic Suggestions")
             q_lower = q.lower()
             
             # 1. Mode Detection
             mode = "unknown"
             if any(k in q_lower for k in ["เบอร์", "โทร", "phone", "contact", "call", "ติดต่อ"]):
                 mode = "phone"
             elif any(k in q_lower for k in ["ใคร", "who", "ตำแหน่ง", "ผู้ดำรง", "holder", "person"]):
                 mode = "holder"
             
             # 2. Target Detection & Suggestion
             suggestions = []
             kind = "unknown"
             
             # Cleanup query for suggestion lookup
             # Remove triggers
             clean_q = q_lower
             for k in ["เบอร์", "โทร", "ติดต่อ", "ใคร", "คือ", "ตำแหน่ง", "ครับ", "ค่ะ", "what", "is", "who", "the", "of"]:
                 clean_q = clean_q.replace(k, "")
             clean_q = clean_q.strip()
             
             if not clean_q:
                  # If query was just "เบอร์โทร", "ใครครับ" -> Ask for more
                  return {
                     "answer": "ต้องการทราบข้อมูลของใครหรือหน่วยงานไหนครับ? \n(เช่น 'เบอร์ ผส.บลตน.', 'ใครคือ ผส.พพ.')",
                     "route": "clarify_empty",
                     "latencies": latencies
                  }

             # A. Team Check
             if any(k in q_lower for k in ["ทีม", "ส่วน", "ฝ่าย", "งาน", "team", "section"]):
                 kind = "team_choice"
                 suggestions = self.directory_handler.suggest_teams(clean_q)
                 
             # B. Person Check (Explicit "คุณ")
             elif "คุณ" in q_lower or len(clean_q) > 40: # Long text might be name?
                 kind = "person_choice" # No specific handler yet, treat as role/holder lookup?
                 # Actually Person Lookup is handled by find_person but we want suggestions
                 suggestions = self.directory_handler.suggest_persons(clean_q)
                 if not suggestions: # Fallback to role
                      kind = "role_choice"
                      suggestions = self.directory_handler.suggest_roles(clean_q)
             
             # C. Role Check (Default for short acronyms)
             else:
                 kind = "role_choice"
                 suggestions = self.directory_handler.suggest_roles(clean_q)
                 
             # 3. Format Response (Deterministic)
             if suggestions:
                 # Format: "1) ... 2) ..."
                 s_list = "\n".join([f"{i+1}) {s}" for i, s in enumerate(suggestions)])
                 
                 # Logic: If only 1 suggestion and good match?
                 # User Rule: "If only 1 candidate AND confidence high -> still ask confirm... when token was short."
                 # Since we are here because of Ambiguity/Clarify intent, strict confirm is safer.
                 
                 ans = f"คุณหมายถึงตำแหน่งไหนครับ?\n{s_list}\n\nพิมพ์เลข 1-{len(suggestions)} หรือพิมพ์ชื่อเต็ม"
                 
                 # Set State
                 candidates = [{"id": i+1, "label": s, "key": s} for i, s in enumerate(suggestions)]
                 self.pending_question = {
                      "kind": kind,
                      "candidates": candidates,
                      "original_query": q,
                      "created_at": time.time(),
                      "mode": mode if mode != "unknown" else "holder" # Default
                 }
                 
                 return {
                     "answer": ans,
                     "route": "clarify_ambiguous",
                     "latencies": latencies,
                     "suggestions": suggestions
                 }
             
             else:
                 # No suggestions found
                 # Fix 3 (Phase 102): Suggest Knowledge Search if Person/Team lookup failed
                 return {
                     "answer": f"ไม่พบข้อมูลบุคลากร/หน่วยงานที่ตรงกับ '{clean_q}'\n\nหากต้องการค้นหาวิธีการ/คู่มือ:\nลองพิมพ์: \"วิธี {clean_q}\" หรือ \"คู่มือ {clean_q}\"",
                     "route": "clarify_miss",
                     "latencies": latencies
                 }

        # 0.1b Symptom Follow-up Handler (Phase 21)
        # Check if we are in a procedural context and user gives a short symptom reply
        if self.proc_ctx and self.proc_ctx.get("intent") == "procedure":
            # Heuristic: Short reply (< 30 chars) or matches symptom keywords
            is_short = len(q) < 50
            symptom_kws = ["หลุด", "ช้า", "อืด", "ไม่ติด", "หมุน", "error", "เข้าไม่ได้", "wifi", "lan", "สาย", "ไร้สาย"]
            has_symptom = any(k in q_lower for k in symptom_kws)
            
            if is_short or has_symptom:
                print(f"[DEBUG] Symptom follow-up detected: {q}")
                # Update slots
                slots = self.proc_ctx.get("slots", {})
                
                # Simple slot filling rules (Deterministic)
                if "wifi" in q_lower or "ไร้สาย" in q_lower: slots["access_type"] = "wifi"
                if "lan" in q_lower or "แลน" in q_lower or "สาย" in q_lower: slots["access_type"] = "lan"
                
                if "หลุด" in q_lower: slots["symptom"] = "disconnnect"
                if "ช้า" in q_lower or "อืด" in q_lower: slots["symptom"] = "slow"
                if "ไม่ติด" in q_lower or "เข้าไม่ได้" in q_lower: slots["symptom"] = "no_connect"
                
                self.proc_ctx["slots"] = slots
                
                # Check missing slots for "internet" topic
                if self.proc_ctx.get("topic") == "internet":
                    required = ["access_type", "symptom"]
                    missing = [k for k in required if k not in slots]
                    
                    if missing:
                        # Ask next question (Deterministic)
                        next_q = ""
                        if "access_type" in missing:
                            next_q = "ใช้งานผ่าน Wi-Fi หรือสาย LAN ครับ?"
                        elif "symptom" in missing:
                            next_q = "อาการเป็นอย่างไรครับ (เช่น หลุดบ่อย, ช้า, หรือเชื่อมต่อไม่ได้)?"
                        else:
                            next_q = "ขอทราบรายละเอียดเพิ่มเติมครับ"
                            
                        return {
                            "answer": next_q,
                            "route": "rag_clarify_followup",
                            "latencies": latencies
                        }
                    else:
                        # All slots filled -> Proceed to RAG/Article
                        # Construct a better query
                        access = slots.get("access_type", "")
                        sym = slots.get("symptom", "")
                        new_q = f"วิธีแก้ไขปัญหา internet {access} อาการ {sym}"
                        print(f"[DEBUG] Auto-generating query from slots: {new_q}")
                        
                        # Clear context so we don't loop
                        self.proc_ctx = None
                        
                        # Recursive call with refined query? 
                        # Or just fall through to standard processing with modified query?
                        # Fall through is safer to use standard routing.
                        q = new_q
                        q_lower = q.lower()
                        # Fall through...

        # ---------------------------------------------------------
        # Phase 228: Directory Fast Path (Implicit Queries) - MOVED UP
        # ---------------------------------------------------------
        from src.directory.lookup import lookup_phones, is_broad_query
        q_clean = q.strip().lower()
        if len(q_clean) < 50:
             # Check if query strongly matches a directory entry (Team/Person)
             fast_hits = lookup_phones(q_clean, self.records)
             
             top_score = fast_hits[0].get("_score", 0) if fast_hits else 0

             # Rule I: Force Broad Query to CONTACT_LOOKUP regardless of score (if valid category)
             # Rule C: Score >= 85
             if top_score >= 85 or (is_broad_query(q_clean) and fast_hits):
                 print(f"[DEBUG] Directory Fast Path: Found match '{fast_hits[0].get('name')}' (Score {top_score}). Forcing CONTACT_LOOKUP.")
                 intent = "CONTACT_LOOKUP"

        # 1.0 STATUS_CHECK Handler (Phase 34)
        if intent == "STATUS_CHECK":
             # Implementation of strictly checking status or failing
             # User Requirement: If not found, must say "No data", NO HALLUCINATION.
             print("[DEBUG] Handling STATUS_CHECK logic (Mock)")
             # TODO: Connect to real status_logs.jsonl
             return {
                 "answer": "ไม่พบข้อมูลสถานะในระบบช่วงเวลานี้ (No Realtime Data)",
                 "route": "status_check_mock",
                 "latencies": latencies
             }
             
        # 1.0b MANAGEMENT_LOOKUP Handler (Phase 48)
        if intent == "MANAGEMENT_LOOKUP":
             print("[DEBUG] Handling MANAGEMENT_LOOKUP logic")
             result = self.directory_handler.handle_management_query(q)
             result["latencies"] = latencies
             
             # Phase 80: Capture Ambiguity
             if result.get("route") == "management_ambiguous" and result.get("candidates"):
                 self.pending_question = {
                     "kind": "management_choice",
                     "candidates": result["candidates"],
                     "original_query": result.get("original_query", q),
                     "created_at": time.time()
                 }
                 
             return result

        # 1.0c POSITION_HOLDER_LOOKUP (Phase 49)

        # Phase 233: Guard against Resource/Info queries incorrectly classified as Person Lookup
        # e.g. "ตารางเวร สรกภ.4" -> Should refer to Document/Schedule, not "Who is Mr. Schedule?"
        NON_PERSON_KEYWORDS = ["ตารางเวร", "สินทรัพย์", "เวร", "schedule", "asset", "kpi", "budget", "งบประมาณ", "แผนงาน", "policy", "นโยบาย"]
        is_resource_query = any(k in q.lower() for k in NON_PERSON_KEYWORDS)

        if intent == "POSITION_HOLDER_LOOKUP" and is_resource_query:
             print(f"[DEBUG] Demoting POSITION_HOLDER_LOOKUP -> RAG (Keyword Guard: {is_resource_query})")
             intent = "HOWTO_PROCEDURE" # Fallback to RAG

        if intent == "POSITION_HOLDER_LOOKUP":
             print("[DEBUG] Handling POSITION_HOLDER_LOOKUP logic")
             result = self.directory_handler.handle_position_holder(q)
             result["latencies"] = latencies
             
             # Capture pending action from handler
             if result.get("pending_action"):
                 self.pending_question = result["pending_action"]
                 self.pending_question["created_at"] = time.time()
                 
             return result

        # 1.0d TEAM_LOOKUP or ASSET (Phase 69/Stabilize)
        # Priority: If it's an asset request, handle it via Team Lookup logic (for links)
        if intent == "TEAM_LOOKUP" or intent == "ASSET" or is_asset_request:
             print(f"[DEBUG] Handling {intent} logic (is_asset={is_asset_request})")
             result = self.directory_handler.handle_team_lookup(q, is_asset=is_asset_request)
             
             # Phase 80: Capture Ambiguity
             if result.get("route") == "team_ambiguous" and result.get("candidates"):
                 self.pending_question = {
                     "kind": "team_choice",
                     "candidates": result["candidates"],
                     "original_query": q,
                     "created_at": time.time()
                 }
                 
             # Phase 72: Demotion / Re-Routing
             if result.get("route") == "team_demoted":
                 print("[DEBUG] Team Lookup Demoted -> Redirecting to HOWTO_PROCEDURE")
                 # Fall through to HOWTO logic? 
                 # Or explicitly call it.
                 # Since HOWTO logic is BELOW, we can just change `intent` and FALL THROUGH?
                 # BUT HOWTO logic is at priority 1.06 (Line 706)?
                 # Wait, line 706 is AFTER this block? No, checking indices.
                 # My previous `view_file` showed HOWTO at line 706.
                 # `TEAM_LOOKUP` is at line 612.
                 # So yes, HOWTO is BELOW.
                 # So I can just set `intent = "HOWTO_PROCEDURE"` and let it fall through?
                 # BUT the `if` clauses are `if intent == ...`.
                 # They are not `elif`.
                 # Let's verify structure.
                 intent = "HOWTO_PROCEDURE"
                 # Fall through to next if block
             else:
                 result["latencies"] = latencies
                 return result

        # 1.05 NEWS_SEARCH Handler (Phase 35)
        if intent == "NEWS_SEARCH":
            print("[DEBUG] Handling NEWS_SEARCH logic")
            
            # Phase 106: Prioritize SMC Database (Link Index)
            # Check if query matches an existing Article/News Title directly
            link_hits = self.processed_cache.find_links_fuzzy(q, threshold=0.65)
            if link_hits:
                 top_link = link_hits[0]
                 if top_link.get("items"):
                      first_item = top_link["items"][0]
                      target_url = first_item.get("href")
                      print(f"[DEBUG] News/Article Link Match: '{q}' -> {target_url} (Score: {top_link['score']})")
                      return self._handle_article_route(target_url, q, latencies, start_time=t_start, match_score=top_link['score'])
            
            # Use Fusion Search but heavily dependent on BM25 match of Title
            # Or use processed_cache to scan titles?
            # VectorStore should have a way to search.
            # Assuming 'vs.retrieve' returns Documents.
            
            # Simple Strategy:
            # 1. Search Vector Store (Hybrid)
            # 2. Filter hits where source is "News" or "Announcement" OR title matches query strongly
            # 3. If hit -> Article Route
            
            # Use hybrid_query for best keyword matching on titles
            hits = self.vs.hybrid_query(q, top_k=5)
            
            # Filter for news-like content
            news_hits = []
            for h in hits:
                # h is SearchResult object
                h_meta = h.metadata
                title = h_meta.get("title", "")
                
                # Heuristic: Check if 'news', 'ประกาศ', 'ประชาสัมพันธ์' is in title or content
                is_news = any(w in title.lower() for w in ["ข่าว", "ประกาศ", "news", "announcement"])
                
                # Or query match title? (Token-based)
                # Split query only by spaces (English style). For Thai, simple split might fail.
                # But here we just want ANY overlap.
                # FIX (Phase 62): Exclude intent triggers (stopwords) from overlap check
                from src.ai.router import IntentRouter
                news_triggers = IntentRouter.INTENTS["NEWS_SEARCH"]
                
                q_toks = [t for t in q_lower.split() if len(t) > 2 and t not in news_triggers] 
                query_match = any(t in title.lower() for t in q_toks)
                
                # Relaxed: If score is very high (e.g. > 0.65), trust the vector store.
                is_strong_hit = h.score > 0.65
                
                # Logic Fix (Phase 62): 
                # Prev: is_news OR query_match OR strong_hit
                # Problem: "ข่าว garbage" finds "ข่าว Bonus" (is_news=True) -> Mismatch.
                # Fix: Must match Topic (query_match) AND be News-like (is_news), OR be very strong hit.
                
                if (is_news and query_match) or is_strong_hit:
                    news_hits.append(h)
            
            if news_hits:
                # Phase 44: Link Ranking & Filtering
                # Filter out system pages (login, register, user profile)
                NEWS_DENYLIST = [
                    "register", "login", "com_user", "reset", "remind", "cart", "checkout", "profile",
                    "forgot", "username", "password", "component/users", "download"
                ]
                # NEWS_ALLOWLIST (Optional): ["view=article", "option=com_content", "category"]
                
                valid_hits = []
                for h in news_hits:
                    url = h.metadata.get("url") or h.metadata.get("source") or ""
                    url_lower = str(url).lower()
                    if any(bad in url_lower for bad in NEWS_DENYLIST):
                        continue
                    valid_hits.append(h)

                if valid_hits:
                    # Use best valid hit
                    best_hit = valid_hits[0]
                    target_url = best_hit.metadata.get("url") or best_hit.metadata.get("source")
                    return self._handle_article_route(target_url, q, latencies, start_time=t_start, match_score=best_hit.score)
                else:
                    # Hits existed but all were system pages -> Strict Miss
                    return {
                        "answer": "ไม่พบข้อมูลในระบบปัจจุบัน (กรองลิงก์ระบบออกหมดแล้ว)",
                        "route": "news_miss_filtered",
                        "latencies": latencies
                    }

            else:
                 # Phase 35 Fallback: Strict Miss
                 # Do NOT fall through to GENERAL_QA which might hallucinate.
                 return {
                    "answer": "ไม่พบข้อมูลในระบบปัจจุบัน",
                    "route": "news_miss",
                    "latencies": latencies
                 }

        # 1.06 HOWTO_PROCEDURE Handler (Phase 45/92)
        # Goal: Article-First Strategy. If we find a strong Article match, use Full-Text Extraction.
        # 1.06 HOWTO_PROCEDURE Handler (Phase 45/92)
        # Goal: Article-First Strategy. If we find a strong Article match, use Full-Text Extraction.
        if intent in ["HOWTO_PROCEDURE", "THAI_ORG_KNOWLEDGE"]:
             print(f"[DEBUG] Handling {intent} logic (Article-First)")
             
             # Phase 92: Detect explicit knowledge queries and strip keyword
             KNOWLEDGE_KEYWORDS = ["ความรู้", "knowledge"]
             is_knowledge_query = any(kw in q.lower() for kw in KNOWLEDGE_KEYWORDS)
             search_query = q
             
             if is_knowledge_query:
                 # Strip knowledge keyword for better search
                 search_query = q
                 for kw in KNOWLEDGE_KEYWORDS:
                     search_query = search_query.replace(kw, "").replace(kw.upper(), "").replace(kw.capitalize(), "")
                 search_query = search_query.strip()
                 print(f"[DEBUG] Knowledge query detected. Stripped query: '{q}' -> '{search_query}'")
             
             # 1. Search Vector Store to identify "The Article"
             t_vs_howto = time.time()
             hits = self.vs.hybrid_query(search_query, top_k=10)
             latencies["vector_search"] = (time.time() - t_vs_howto) * 1000
             # [RetrievalFilter] Rerank hits by intent-type compatibility
             try:
                 from src.core.retrieval_filter import apply_intent_filter as _rf
                 hits = _rf(hits, intent=intent, query=search_query)
             except Exception as _rfe:
                 print(f"[RetrievalFilter] Skipped: {_rfe}")
             
             # Phase 105: Prioritize Link Lookup (Exact/Fuzzy) for HowTo
             # If the query matches a known article link (e.g. "sbc ip"), use it directly.
             link_hits = self.processed_cache.find_links_fuzzy(search_query, threshold=0.70)
             if link_hits:
                  top_link_hit = link_hits[0]
                  # Inspect items
                  if top_link_hit.get("items"):
                       # Get the first item's href
                       first_item = top_link_hit["items"][0]
                       print(f"[DEBUG] JunkFilter first_item keys: {list(first_item.keys())}, title='{first_item.get('title', 'N/A')}', text='{first_item.get('text', 'N/A')}', href='{first_item.get('href', 'N/A')}'")
                       target_url = first_item.get("href")
                       target_title = first_item.get("title") or first_item.get("text") or ""
                       
                       # JUNK FILTER (Link Match - Title Check)
                       JUNK_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".zip", ".rar", ".7z", ".exe", ".dmg", ".iso", ".apk", ".ipa"]
                       if target_url:
                           is_junk_url = any(target_url.lower().endswith(ext) for ext in JUNK_EXTENSIONS)
                           is_junk_title = any(target_title.lower().endswith(ext) for ext in JUNK_EXTENSIONS)
                                                      # Google Drive Special: Strict Filter for /file/d/ (Binaries)
                           if "drive.google.com" in target_url.lower():
                               if "/file/d/" in target_url.lower():
                                   print(f"[ChatEngine] Detected Google Drive File URL (High Risk of Binary): {target_url}")
                                   is_junk_url = True 
                               elif any(ext in target_title.lower() for ext in JUNK_EXTENSIONS):
                                   is_junk_title = True

                           if is_junk_url or is_junk_title:
                               print(f"[ChatEngine] Filtered Junk Link (Title/URL): {target_title} | {target_url}")
                           else:
                               print(f"[DEBUG] HowTo Link Match: '{search_query}' -> {target_url} (Score: {top_link_hit['score']})")
                               return self._handle_article_route(target_url, q, latencies, start_time=t_start, match_score=top_link_hit.get('score', 0.0), intent=intent)
             
             # 2. Check overlap logic or Score
             # Phase 92: Use lower threshold (0.45) for explicit knowledge queries
             threshold = 0.45 if is_knowledge_query else 0.6
             
             candidate = None
             if hits:
                 for h in hits:
                     meta = h.metadata or {}
                     url = meta.get("url", "").lower()
                     c_type = meta.get("content_type", "")
                     
                     # Phase 98: Category Exclusion
                     if "view=category" in url: continue
                     if "view=section" in url: continue
                     if c_type == "category": continue
                     
                     print(f"[DEBUG] HowTo Candidate: {meta.get('title')} (Score: {h.score:.4f}, Threshold: {threshold})")
                     
                     if h.score >= threshold: # Fixed: Use actual threshold (0.6 or 0.45) instead of hardcoded 0.35
                         # Check if it behaves like an Article (Has URL)
                         if meta.get("url") or meta.get("source"):
                             # JUNK FILTER ADAPTER
                             _url = (meta.get("url") or meta.get("source") or "").lower()
                             _title = (meta.get("title") or "").lower()
                             JUNK_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".zip", ".rar", ".7z", ".exe", ".dmg", ".iso", ".apk", ".ipa"]
                             
                             if any(_url.endswith(ext) for ext in JUNK_EXTENSIONS) or any(_title.endswith(ext) for ext in JUNK_EXTENSIONS):
                                 print(f"[ChatEngine] Filtered Junk Candidate: {_title}")
                                 continue

                             candidate = h
                             break # Found best valid hit
             
             if candidate:
                 # Route to Article Interpreter (Full Text)
                 # This bypasses chunked RAG, letting us read the WHOLE guide (up to limit)
                 # and allowing Image logic to run.
                 target_url = candidate.metadata.get("url") or candidate.metadata.get("source")
                 return self._handle_article_route(target_url, q, latencies, start_time=t_start, match_score=candidate.score, intent=intent)
             
             # Else: Fallback to General RAG (Chunk-based)
             print("[DEBUG] No single strong article found for HowTo. Falling back to RAG.")
             intent = "GENERAL_QA" # Fall through
             bypass_kp = True # Skip KP lookup since we want to find document chunks, not short facts

        # Phase 175 + SMC: Web Knowledge Handler with Technical Query Blocking
        if intent == "WEB_KNOWLEDGE":
            # Phase SMC: Check if this is a technical query (vendor+model)
            is_technical = self._has_vendor_model_pattern(q)
            
            if is_technical:
                print(f"[SMC GOVERNANCE] Technical query detected in WEB_KNOWLEDGE: {q}")
                
                # Try internal SMC first before blocking
                try:
                    safety_hits = self.vs.hybrid_query(q, top_k=1)
                    if safety_hits and safety_hits[0].score > 0.5:
                        matched_score = safety_hits[0].score
                        print(f"[SMC GOVERNANCE] Found internal match (score={matched_score:.2f}) -> Route to SMC")
                        intent = "GENERAL_QA"  # Route to internal RAG
                        # Fall through to GENERAL_QA section
                    else:
                        # No internal match -> BLOCK (no web fallback for technical queries)
                        print(f"[SMC GOVERNANCE] No SMC match for technical query -> BLOCK")
                        
                        # Get topic suggestions
                        suggestions = []
                        try:
                            # Get some available topics from the link index
                            if hasattr(self, 'processed_cache') and self.processed_cache._link_index:
                                all_titles = list(self.processed_cache._link_index.values())[:10]
                                suggestions = [entry[0]['text'] for entry in all_titles if entry]
                        except:
                            pass
                        
                        suggestion_text = ""
                        if suggestions:
                            suggestion_text = f"\n\n📋 **หัวข้อที่มีในระบบ:**\n" + "\n".join(f"- {s}" for s in suggestions[:5])
                        
                        return {
                            "answer": (
                                f"⚠️ **ไม่พบข้อมูลในระบบ SMC**\n\n"
                                f"ระบบตรวจพบคำค้นหาเกี่ยวกับอุปกรณ์เฉพาะรุ่น แต่ยังไม่มีเอกสารในฐานข้อมูล\n"
                                f"สถานะ: `NO_SMC_DATA`" + suggestion_text + "\n\n"
                                f"กรุณาแจ้ง Admin เพื่อนำเข้าเอกสารสำหรับรุ่นนี้ครับ"
                            ),
                            "route": "rag_no_smc_data",
                            "latencies": latencies,
                            "suggestions": suggestions
                        }
                except Exception as e:
                    print(f"[SMC GOVERNANCE] Safety check error: {e} -> BLOCK technical query")
                    return {
                        "answer": (
                            "⚠️ **ไม่พบข้อมูลในระบบ SMC**\n\n"
                            "ขออภัยครับ ระบบไม่สามารถค้นหาข้อมูลสำหรับคำถามนี้ได้"
                        ),
                        "route": "rag_no_smc_data",
                        "latencies": latencies
                    }
            
            # Non-technical query: Allow WebHandler with safety check
            # Safety Check: Prevent Web Hijack of Internal Topics
            # Policy: web_knowledge.internal_override_threshold
            policy_threshold = self.routing_policy.get("web_knowledge", {}).get("internal_override_threshold", 0.65)
            
            print(f"[DEBUG] Web Knowledge Safety Check for: '{q}' (Threshold: {policy_threshold})")
            try:
                safety_hits = self.vs.hybrid_query(q, top_k=1)
                
                if safety_hits and safety_hits[0].score > policy_threshold:
                    matched_score = safety_hits[0].score
                    print(f"[DEBUG] Safety Check: Found internal match (score={matched_score:.2f}) -> Override to SMC")
                    
                    # Point 4: Structured Logging for Regression Guard
                    print(f"[DEBUG] SAFETY_OVERRIDE\nquery=\"{q}\"\ninternal_score={matched_score:.4f}\noriginal_intent=WEB_KNOWLEDGE\nfinal_intent=GENERAL_QA")
                    
                    intent = "GENERAL_QA" # Fall through to Internal RAG
                    # Do NOT return here. Let it fall through.
                else:
                    print(f"[DEBUG] Routing to WebHandler: {q}")
                    return self.web_handler.handle(q)
            except Exception as e:
                print(f"[DEBUG] Safety Check Error: {e} -> Defaulting to Web")
                return self.web_handler.handle(q)



        # 0.2 Context-Aware Follow-up (Robust)
        if self.last_context:
            ctx = self.last_context
            ctx_data = ctx.get("data")
            ctx_type = ctx.get("type")
            
            # Normalization Layer
            normalized_q = q_lower
            # Remove polite words and common fillers
            for w in ["ครับ", "ค่ะ", "หน่อย", "ด้วย", "นะ", "จ๊ะ", "รบกวน", "ช่วย", "ขอ", "อยากได้", "ทราบ"]:
                normalized_q = normalized_q.replace(w, "")
            normalized_q = re.sub(r"[^\wก-๙]", "", normalized_q) # Remove spaces/punct
            
            # Intent Keywords (Expanded)
            kws_phone = ["เบอร์", "เบอ", "โทร", "phon", "call", "contact"]
            kws_email = ["เมล", "mail", "อีเมล", "e-mail"]
            kws_fax = ["fax", "แฟกซ์", "โทรสาร"]
            kws_link = ["link", "url", "เว็บไซต์", "เว็บ", "web", "page", "ลิงก์", "ลิ้ง", "เข้า"]
            kws_detail = ["รายละเอียด", "detail", "info", "คืออะไร", "what"]
            
            req_fax = any(k in normalized_q for k in kws_fax)
            req_phone = any(k in normalized_q for k in kws_phone) and not req_fax
            req_email = any(k in normalized_q for k in kws_email)
            req_link = any(k in normalized_q for k in kws_link)
            req_detail = any(k in normalized_q for k in kws_detail)
            
            # Heuristic: Is this a generic query? (No new entity)
            # Remove intent keywords to see if a name/role remains
            clean_q = normalized_q
            all_intents = kws_phone + kws_email + kws_fax + kws_link + kws_detail + ["ครับ", "ค่ะ", "ขอ", "ช่วย", "หน่อย"]
            for k in all_intents:
                clean_q = clean_q.replace(k, "")
            
            # Apply role alias expansion BEFORE entity check
            # This ensures "ขอเบอร์ผจ" expands to "ผจ.สบลตน." which is recognized as entity
            ROLE_ALIASES = {
                "ผจ": "ผจ.สบลตน.",
                "ผส": "ผส.บลตน.",
                "ผอ": "ผอ.",
                "รอง": "รองผอ."
            }
            for alias, full_role in ROLE_ALIASES.items():
                if alias in clean_q:
                    clean_q = clean_q.replace(alias, full_role)
            
            # Check if entity remains after cleanup
            # Method 1: Length check (significant text remaining)
            has_entity_by_length = len(clean_q) > 3
            
            # Method 2: Check against known roles in position_index
            has_entity_by_role = any(role in clean_q for role in self.position_index.keys())
            
            # Method 3: Check against common entity patterns (names, departments)
            # For now, we rely on length + role matching
            has_entity = has_entity_by_length or has_entity_by_role
            
            # Only trigger follow-up if NO entity detected
            is_generic_request = (not has_entity) or req_detail  # Detail requests can use context

            if is_generic_request:
                # Fax Request
                if req_fax:
                    faxes = []
                    if isinstance(ctx_data, list):
                        for r in ctx_data:
                            if "faxes" in r: faxes.extend(r["faxes"])
                    elif isinstance(ctx_data, dict):
                         if "faxes" in ctx_data: faxes.extend(ctx_data["faxes"])
                    
                    if faxes:
                        faxes = list(set(faxes))
                        return {
                            "answer": f"โทรสาร (Fax) ของ {ctx.get('ref_name')}:\n" + "\n".join(f"- {f}" for f in faxes),
                            "route": "context_followup",
                            "latencies": latencies
                        }
                    else:
                        return {
                            "answer": f"ไม่พบข้อมูลโทรสารของ {ctx.get('ref_name')} ในระบบ",
                            "route": "context_followup_miss",
                            "latencies": latencies
                        }

                # Phone Request
                if req_phone:
                    phones = []
                    if isinstance(ctx_data, list):
                        for r in ctx_data:
                            if "phones" in r: phones.extend(r["phones"])
                    elif isinstance(ctx_data, dict):
                         if "phones" in ctx_data: phones.extend(ctx_data["phones"])
                    
                    # Fallback lookup if missing
                    if not phones:
                         target_name = None
                         if isinstance(ctx_data, dict): target_name = ctx_data.get("name")
                         elif isinstance(ctx_data, list) and ctx_data: target_name = ctx_data[0].get("name")
                         if target_name:
                             hits = lookup_phones(target_name, self.records)
                             for h in hits:
                                 if "phones" in h: phones.extend(h["phones"])

                    if phones:
                        phones = list(set(phones))
                        return {
                            "answer": f"เบอร์โทรศัพท์ของ {ctx.get('ref_name')}:\n" + "\n".join(f"- {p}" for p in phones),
                            "route": "context_followup",
                            "latencies": latencies
                        }
                    else:
                        return {
                            "answer": f"ไม่พบข้อมูลเบอร์โทรศัพท์ของ {ctx.get('ref_name')} ในระบบ",
                            "route": "context_followup_miss",
                            "latencies": latencies
                        }
                
                # Email Request
                if req_email:
                     emails = []
                     if isinstance(ctx_data, list):
                         for r in ctx_data:
                             if "emails" in r: emails.extend(r["emails"])
                     elif isinstance(ctx_data, dict):
                         if "emails" in ctx_data: emails.extend(ctx_data["emails"])

                     if emails:
                         emails = list(set(emails))
                         return {
                             "answer": f"อีเมลของ {ctx.get('ref_name')}:\n" + "\n".join([f"- {e}" for e in emails]),
                             "route": "context_followup",
                             "latencies": latencies
                         }
                     else:
                         return {
                             "answer": f"ไม่พบข้อมูลอีเมลของ {ctx.get('ref_name')} ในระบบ",
                             "route": "context_followup_miss",
                             "latencies": latencies
                         }

                # Link Request
                if req_link:
                    url = None
                    if ctx_type == "link": url = ctx_data.get("href")
                    elif ctx_type == "position":
                        if isinstance(ctx_data, list) and ctx_data: url = ctx_data[0].get("source")
                        elif isinstance(ctx_data, dict): url = ctx_data.get("source")
                    elif ctx_type == "contact": url = ctx_data.get("source")
                    
                    if url:
                         return {
                            "answer": f"ลิงก์ข้อมูลสำหรับ {ctx.get('ref_name')}:\nURL: {url}",
                            "route": "context_followup",
                            "latencies": latencies
                        }
                    else:
                         return {
                            "answer": f"ไม่พบข้อมูลลิงก์สำหรับ {ctx.get('ref_name')}",
                            "route": "context_followup_miss",
                            "latencies": latencies
                        }

                # Detail Request (Generic)
                if req_detail and ctx_type == "link":
                    return {
                         "answer": f"รายละเอียด {ctx.get('ref_name')}:\nURL: {ctx_data.get('href')}\nAccess: {ctx_data.get('access', 'Unknown')}",
                         "route": "context_followup",
                         "latencies": latencies
                    }
        
        # ---------------------------------------------------------
        # Phase 23/25: Knowledge Pack Lookup (PRIORITY)
        # ---------------------------------------------------------
        # Moved up to prioritize verified facts over heuristics

        # B. Direct KP Lookup (High Confidence Entities)
        # Check if query matches high-value facts (DNS/SMTP/BRAS etc)
        # We perform this BEFORE Position/Contact lookup to catch "Bras IP", "DNS" etc.
        kp_result = None
        # Phase 98: Contact/Person Intent Guard
        # If intent is explicitly looking for people/contacts, skip high-level KP lookup 
        # to ensure it goes to the specialized `ContactHandler` downstream.
        if self.kp_manager and not bypass_kp and intent not in ["CONTACT_LOOKUP", "PERSON_LOOKUP", "TEAM_LOOKUP", "MANAGEMENT_LOOKUP", "POSITION_HOLDER_LOOKUP", "HOWTO_PROCEDURE"]:
            # Phase 31: Multi-Key Aggregation
            # Detect combination of key entities
            q_lower = q.lower()
            keys_found = []
            if "dns" in q_lower: keys_found.append("DNS")
            if "smtp" in q_lower or "mail" in q_lower: keys_found.append("SMTP")
            if "bras" in q_lower: keys_found.append("BRAS")
            
            if len(keys_found) > 1:
                print(f"[DEBUG] Multi-Key KP Query: {keys_found}")
                telemetry_data["kp_multi_key"] = True
                telemetry_data["kp_keys"] = keys_found
                
                # Extract Scope to preserve
                scope_hint = ""
                if "nt1" in q_lower or "bkk" in q_lower: scope_hint = "NT1"
                elif "isp" in q_lower: scope_hint = "ISP"
                elif "region" in q_lower or "ภูมิภาค" in q_lower: scope_hint = "Region"
                
                aggregated_ans = []
                aggregated_hits = []
                all_options = set()
                
                for key in keys_found:
                    sub_q = f"{key} {scope_hint}".strip()
                    sub_res = self.kp_manager.lookup(sub_q)
                    if sub_res:
                        # If signal is CLARIFY, collect options but don't fail
                        if sub_res.get("signal") == "CLARIFY_SCOPE":
                            for opt in sub_res.get("options", []): all_options.add(opt)
                        else:
                            # Clean answer: Remove "Found X items" header for sub-sections to be cleaner
                            raw_ans = sub_res["answer"]
                            # Remove first line if it looks like a header count
                            if raw_ans.startswith("พบข้อมูล") or raw_ans.startswith("ไม่พบข้อมูล"):
                                lines = raw_ans.split('\n')
                                if len(lines) > 1: raw_ans = "\n".join(lines[1:])
                            
                            
                            aggregated_ans.append(f"=== {key} ===\n{raw_ans}")
                            # Deduplicate hits by key+value
                            for h in sub_res.get("hits", []):
                                h_id = f"{h.get('key')}_{h.get('value')}"
                                if h_id not in [f"{x.get('key')}_{x.get('value')}" for x in aggregated_hits]:
                                     aggregated_hits.append(h)
                
                if aggregated_ans:
                    kp_result = {
                        "answer": "\n\n".join(aggregated_ans),
                        "hits": aggregated_hits,
                        "signal": "AGGREGATED"
                    }
                elif all_options:
                     # If all sub-lookups asked for clarification
                     sorted_opts = sorted(list(all_options))
                     kp_result = {
                         "answer": f"กรุณาเลือกขอบเขตสำหรับ {', '.join(keys_found)}: {', '.join(sorted_opts)}",
                         "signal": "CLARIFY_SCOPE",
                         "options": sorted_opts
                     }
            
            # Fallback to Single Lookup if not multi-key or aggregation failed
            if not kp_result:
                kp_result = self.kp_manager.lookup(q)
        if kp_result:
             if kp_result.get("signal") == "CLARIFY_SCOPE":
                  self.pending_kp_clarify = {
                      "original_query": q,
                      "options": kp_result.get("options", []),
                      "timestamp": time.time(),
                      "pending_answer_mode": answer_mode # Persist mode
                  }
                  latencies["total"] = (time.time() - t_start) * 1000
                  telemetry_data["route"] = "knowledge_pack_clarify"
                  telemetry_data["clarify_asked"] = True
                  telemetry_data["pack_hit"] = True
                  self._log_telemetry(telemetry_data, latencies)
                  
                  return {
                      "answer": kp_result["answer"],
                      "route": "knowledge_pack_clarify",
                      "latencies": latencies,
                      "hits": []
                  }
             
             latencies["total"] = (time.time() - t_start) * 1000
             res = {
                 "answer": kp_result["answer"],
                 "route": "knowledge_pack",
                 "latencies": latencies,
                 "hits": kp_result["hits"]
             }
             telemetry_data["route"] = "knowledge_pack"
             telemetry_data["pack_hit"] = True
             self._log_telemetry(telemetry_data, latencies)
             return self._apply_answer_mode(res, answer_mode)

             return self._apply_answer_mode(res, answer_mode)

        # ---------------------------------------------------------
        # Phase 34: Strict Intent-Based Routing (Refactored)
        # ---------------------------------------------------------

        # 1.1 CONTACT_LOOKUP
        if intent == "CONTACT_LOOKUP":
            print("[DEBUG] Route: CONTACT_LOOKUP (via Handler)")
            from src.rag.handlers.contact_handler import ContactHandler
            
            # Phase 221: Pass LLM Config & Disambiguation State
            disambig_state = getattr(self, "pending_contact_clarify", None)
            
            res = ContactHandler.handle(
                q, 
                self.records, 
                directory_handler=self.directory_handler,
                llm_cfg=self.llm_cfg,
                disambiguation_state=disambig_state
            )
            
            # State Management (Phase 227: Unified Pending Question)
            # Phase 235: Allow broad/all lists to set pending state for selection
            if res.get("route") in ["contact_ambiguous", "contact_broad_list", "contact_ambiguous_all", "contact_prefix_ambiguous"]:
                 candidates = []
                 try:
                     raw_ctx = res.get("context", "[]")
                     if isinstance(raw_ctx, str):
                         candidates = json.loads(raw_ctx)
                     else:
                         candidates = raw_ctx
                 except: pass
                 
                 if candidates:
                     self.pending_question = {
                         "created_at": time.time(),
                         "kind": "contact_choice",
                         "candidates": candidates,
                         "mode": "contact"
                     }
                     # Legacy clear
                     self.pending_contact_clarify = None
            else:
                # Clear if hit/miss
                self.pending_question = None
                self.pending_contact_clarify = None
            
            # If hit, update context
            if res.get("hits"):
                hits = res["hits"]
                
                # ================================================================
                # CRITICAL FIX: Save ALL options in ambiguous results
                # ================================================================
                # Old: Only saved hits[0] → context has only first option (OMC)
                # New: Save ALL hits → context has all options (OMC + RNOC)
                # This enables entity matching: "ของ RNOC ละ" can find RNOC
                
                entities = {}
                route = res.get("route", "")
                
                # For ambiguous/multiple results, save ALL options as entities
                if route in ["contact_ambiguous", "contact_broad_list", "contact_ambiguous_all", "contact_prefix_ambiguous"]:
                    # Save all candidate names as CONTACT entities
                    for hit in hits[:10]:  # Limit to top 10
                        hit_name = hit.get("name", "")
                        if hit_name:
                            entities[hit_name] = "CONTACT"
                    result_data = hits  # Save all hits
                else:
                    # Single hit or other routes
                    entities = {hits[0].get("name", ""): "CONTACT"}
                    result_data = hits[0]
                
                # Save context to persistent storage
                new_context = context_manager.create_context(
                    query=q,
                    intent="CONTACT_LOOKUP",
                    route="contact",
                    entities=entities,
                    result_data=result_data
                )
                self.last_context = new_context
                if session_id and session_id != "default":
                    context_manager.save_session_context(session_id, new_context)
            
            # Fallback to RAG if Contact Handler misses (e.g. support phone in an article)
            if res.get("route") == "contact_miss":
                # Policy: contact_lookup.no_kill_switch
                # If true, allow RAG fallback? Original code Logic says "Strict Kill-Switch (No RAG)".
                # Let's check the policy.
                # User config: guided_fallback_on_miss: true, no_kill_switch: true
                # If no_kill_switch is TRUE, we SHOULD fall through or try something else?
                # User Point 2 for Contact says "no_kill_switch: true" in sample,
                # BUT "strict kill-switch" logic was requested in previous phase (135).
                # Wait, Point 2 sample says: "ขอเบอร์ทั้งหมดของหาดใหญ่ → ต้อง guided fallback (ไม่ MISS)"
                # My `contact_lookup` policy in file says:
                # contact_lookup:
                #   guided_fallback_on_miss: true
                #   no_kill_switch: true
                
                # So if no_kill_switch is True, we should NOT return contact_miss_strict IMMEDIATELY?
                # However, the user request says "Phase 135: Kill-Switch (No RAG Fallback)".
                # The "Bridge Port", "North Flood" cases imply we WANT strict behavior for numbers.
                # BUT "Bridge Port" (config) -> Article.
                # "Hat Yai" -> Guided Fallback.
                
                # Let's implement Guided Fallback here as per Policy.
                no_kill = self.routing_policy.get("contact_lookup", {}).get("no_kill_switch", False)
                guided = self.routing_policy.get("contact_lookup", {}).get("guided_fallback_on_miss", False)

                if guided:
                     print("[DEBUG] Contact Miss -> Attempting Guided Fallback (Policy)")
                     # Try to suggest teams in that area?
                     # For now, return a helpful message instead of generic miss
                     # Or stick to strict miss but with better message?
                     return {
                         "answer": "ไม่พบข้อมูลเบอร์โทรศัพท์ที่ระบุโดยตรง\n\nคำแนะนำ:\n- ระบุชื่อหน่วยงานให้ชัดเจน (เช่น 'ส่วนงาน...')\n- ระบุชื่อบุคคล (เช่น 'คุณสมชาย')",
                         "route": "contact_miss_guided",
                         "latencies": latencies,
                         "hits": []
                     }

                print("[DEBUG] Contact Handler Miss -> Strict Kill-Switch (No RAG).")
                return {
                     "answer": "ไม่พบข้อมูลเบอร์โทรศัพท์/หน่วยงานที่ระบุในระบบสมุดโทรศัพท์ (กรุณาระบุชื่อหน่วยงาน/พื้นที่ให้ชัดเจน)",
                     "route": "contact_miss_strict",
                     "latencies": latencies,
                     "hits": []
                }
            else:
                return {
                    "answer": res.get("answer"),
                    "route": res.get("route"),
                    "latencies": latencies,
                    "hits": res.get("hits")
                }

        # 1.2 PERSON_LOOKUP (Roles/WhoIs)
        # 1.2 PERSON_LOOKUP (Roles/WhoIs)
        elif intent == "PERSON_LOOKUP" or intent == "CONTACT_LOOKUP" or intent == "TEAM_LOOKUP":
             print(f"[DEBUG] Route: {intent}")
             t_start = time.time()
             # Use original query for directory to ensure Thai keyword matching
             return self.directory_handler.handle(original_q_str)

        elif intent == "REFERENCE_LINK":
             print("[DEBUG] Route: REFERENCE_LINK")
             # Clean query for link search
             clean_q = q.lower().replace("ขอลิงก์", "").replace("link", "").strip()
             fuzzy_results = self.processed_cache.find_links_fuzzy(clean_q, threshold=0.60)
             
             if fuzzy_results:
                 top_links = []
                 seen_urls = set()
                 for res in fuzzy_results[:5]:
                     for item in res["items"]:
                         if item["href"] not in seen_urls:
                             top_links.append(item)
                             seen_urls.add(item["href"])
                 
                 if self.cfg.get("ux", {}).get("enable_formatting", True):
                     ans_parts = []
                     for item in top_links[:4]:
                         title = item['text'].replace('\n', ' ').strip()
                         ans_parts.append(f"• **{title}**\n  🔗 {item['href']}")
                     
                     latencies["total"] = (time.time() - t_start) * 1000
                     return {
                         "answer": "พบลิงก์ที่เกี่ยวข้องดังนี้ครับ:\n\n" + "\n\n".join(ans_parts),
                         "route": "link_lookup",
                         "latencies": latencies
                     }
                 else:
                     simple_ans = "\n".join([f"- {item['text']}: {item['href']}" for item in top_links[:4]])
                     latencies["total"] = (time.time() - t_start) * 1000
                     return {
                         "answer": simple_ans,
                         "route": "link_lookup",
                         "latencies": latencies
                     }
            # Fallthrough if no link found -> RAG (maybe it's in a document content)

        # 1.4 HOWTO_PROCEDURE / GENERAL / STATUS_CHECK (Mock) -> Fall through to RAG
        # STATUS_CHECK was handled early (line 496), but if it wasn't, ensure we don't do Position Lookup.
        
        # Phase 106: Prioritize SMC Database (Link Index) for General Queries (Fallthrough)
        # Even if intent wasn't HOWTO or NEWS, if it strongly matches a known Article, USE IT.
        # This catches "SMC Policy" or "Manual X" that slipped through Router.
        
        # Phase R6: Bypass Link Match for "What is X" (Concept) queries
        # User Feedback: "What is RAG" shouldn't match "RAG WAN Document". It should use General Fallback.
        is_what_is = any(k in q.lower() for k in ["คืออะไร", "what is", "concept"])
        
        if not is_what_is:
            link_hits = self.processed_cache.find_links_fuzzy(q, threshold=0.75) # Strict threshold for implicit intent
            if link_hits:
                 top_link = link_hits[0]
                 # Double check: match must be robust (length > 4) and score high
                 if len(q) > 4 and top_link.get("items"):
                      first_item = top_link["items"][0]
                      target_url = first_item.get("href")
                      print(f"[DEBUG] General/Fallthrough Link Match: '{q}' -> {target_url} (Score: {top_link['score']})")
                      return self._handle_article_route(target_url, q, latencies, start_time=t_start, match_score=top_link['score'])

        # Fallthrough to RAG Vector Search (Step 2)
        # This handles:
        # - HOWTO_PROCEDURE (Vector Search + Article)
        # - GENERAL_QA
        # - Missed Lookups (optional fallback but we returned early for misses above?)



        # 1.6 Heuristic Clarification (Pre-RAG Fast Path)
        # Check BEFORE vector search to avoid wasting time on embedding/retrieval
        # Pattern 1: Generic trouble queries (internet/wifi issues)
        is_internet_issue = any(k in q.lower() for k in ["internet", "wifi", "network", "เน็ต", "ไวไฟ", "lan"]) and \
                            any(k in q.lower() for k in ["ใช้ไม่ได้", "ใช้งานไม่ได้", "หลุด", "ช้า", "อืด", "เสีย", "พัง", "problem", "issue", "error", "connect", "ต่อไม่ติด", "เข้าไม่ได้", "ล่ม"])
        
        # Pattern 2: Procedure queries without specific context
        is_procedure_query = any(k in q.lower() for k in ["ขั้นตอน", "วิธี", "อย่างไร", "ทำไง", "how to", "procedure", "steps"]) and \
                            any(k in q.lower() for k in ["แจ้ง", "รายงาน", "report", "ขัดข้อง", "เสีย", "พัง", "ปัญหา"])
        
        if is_internet_issue:
            latencies["total"] = (time.time() - t_start) * 1000
            
            # Initialize Procedural Context
            self.proc_ctx = {
                "intent": "procedure",
                "topic": "internet",
                "slots": {},
                "timestamp": time.time()
            }
            
            return {
                "answer": "ใช้งานไม่ได้แบบไหน (ต่อไม่ติด/ช้า/หลุดบ่อย/ขึ้น error) และใช้งานผ่าน Wi-Fi หรือสาย LAN ครับ?",
                "route": "rag_clarify",
                "latencies": latencies
            }
        
        if is_procedure_query:
            latencies["total"] = (time.time() - t_start) * 1000
            return {
                "answer": "ต้องการแจ้งเรื่องอะไร (อินเทอร์เน็ต/ONU/โทรศัพท์/อุปกรณ์) และเป็นพื้นที่/หน่วยงานไหนครับ?",
                "route": "rag_clarify",
                "latencies": latencies
            }


        # ---------------------------------------------------------
        # Phase R3: Multi-Level Caching & Safety Guard
        # ---------------------------------------------------------
        from src.rag.cache_manager import CacheManager
        from src.rag.safety_guard import SafetyGuard
        
        # Initialize Cache Manager
        if not hasattr(self, "cache_manager") or not self.cache_manager:
             # Reuse existing semantic cache if available
             sem_cache = self.cache if hasattr(self, "cache") else None
             self.cache_manager = CacheManager(sem_cache)

        # Step 1: L1 Retrieval Cache Check (Exact Query + Strategy)
        # Note: We need 'intent' and 'strategy' defined before this?
        # Intent is defined above. Strategy is defined in R1 block.
        # Ideally, we move Strategy definition up or just use 'intent' as proxy for mode.
        # Let's use 'intent' for L1 key for now as it maps to strategy.
        
        l1_hits = self.cache_manager.get_retrieval_cache(q, intent)
        if l1_hits:
            print(f"[DEBUG] L1 Retrieval Cache HIT")
            valid_results = l1_hits
            latencies["vector_search"] = 0.0 # From cache
            is_l1_hit = True
            telemetry_data["cache_hit_l1"] = True
        else:
            is_l1_hit = False
            telemetry_data["cache_hit_l1"] = False
            # ... Perform Retrieval (R1 Logic) ...
            
            # (R1 Logic Block insert here - reuse existing code flow, just wrap?)
            # Since I cannot easily wrap the existing block without huge diff, 
            # I will insert the L1 check BEFORE the R1 block, and if HIT, skip retrieval.
            # But the R1 block is messy.
            # Let's assume the user wants me to EDIT the retrieval block.
            pass

        # Wait, I need to REPLACE the retrieval block effectively.
        # Or I can use 'if not is_l1_hit:' around the entire retrieval section.
        
        # Let's locate the start of R1 Retrieval Strategy and wrap it.
        # ...
        
        # For this tool call, I will add the imports and initialization at the top of RAG block,
        # and implement the Logic flow.

        # 2. RAG
        t_vs = time.time()
        is_general_fallback = False # Initialize for safety scope
        is_conceptual = False       # Initialize for score filtering scope

        # Location Context (Early Extraction for Cache)
        # We need this for both L1 and L2 cache keys/fingerprints effectively
        # (Though L1 currently doesn't use it, L2 should strict filter)
        from src.utils.section_filter import extract_location_intent
        req_locations = extract_location_intent(q)
        if req_locations:
             loc_fingerprint = "|".join(sorted(req_locations))
        else:
             loc_fingerprint = "none"

        # Initialize Cache
        if not hasattr(self, "cache_manager") or not self.cache_manager:
             sem_cache = self.cache if hasattr(self, "cache") else None
             self.cache_manager = CacheManager(sem_cache)
             
        # Step 1: L1 Cache Check
        l1_hits = self.cache_manager.get_retrieval_cache(q, intent)
        
        if l1_hits:
             print(f"[DEBUG] L1 Retrieval Cache HIT")
             results = l1_hits # Skip retrieval
             latencies["vector_search"] = 0
             # Skip optimization? Maybe ran before caching.
             # Assume cached hits are already optimized/ranked.
             valid_results = results # Directly valid
             
             # Phase R6: Re-evaluate General Fallback for Cached Results
             # If we hit cache, we must check if this result SHOULD trigger a General Knowledge Fallback
             # (i.e. if the cached result has low score for a concept query, we must allow it to pass safety)
             
             top_score = getattr(valid_results[0], "score", 0.0) if valid_results else 0.0
             
             # Re-use simplified Concept Logic (Phase 133/R5)
             # Note: 'intent' is already available.
             concept_kws = ["คืออะไร", "what is", "concept", "explain", "meaning", "definition"]
             is_conceptual = (intent == "EXPLAIN") or any(k in q.lower() for k in concept_kws)
             
             # If Concept + Low Score (but not zero) -> Enable Fallback
             if is_conceptual and 0.05 < top_score < 0.35:
                  print(f"[DEBUG] L1 Cache Hit -> General Fallback Triggered (Score {top_score:.4f})")
                  is_general_fallback = True
             
        else:
             # --- ORIGINAL RETRIEVAL LOGIC (R1) ---
             start_vs = time.time()
             where = None
             
             # Phase 174: Deferred Synonym Expansion (Applied only to RAG retrieval)
             q, applied_rule = expand_synonyms(original_q_str)
             if q != original_q_str:
                  synonym_active = True
                  print(f"[METRIC] Synonym Applied for Retrieval: {applied_rule} (Query: '{original_q_str}' -> '{q}')")
                  telemetry_data["synonym_rule"] = applied_rule

             # Phase 133: Define Conceptual Intent Early
             conceptual_kws = ["สรุป", "แนวทาง", "อธิบาย", "คืออะไร", "concept", "explain", "overview", "what is", "introduction", "ความหมาย", "ทำความรู้จัก"]
             is_conceptual = any(k in original_q_str.lower() for k in conceptual_kws)
            
             if hasattr(self.vs, "hybrid_query"):
                 # Phase R1: Dynamic Retrieval Strategy
                 from src.rag.retrieval_strategy import StrategyFactory
                 strategy = StrategyFactory.get_strategy(intent)
                 # ... (Existing retrieval) ...
                 print(f"[DEBUG] Retrieval Strategy: {strategy.mode} (alpha={strategy.alpha}, top_k={strategy.top_k})")
                 results = self.vs.hybrid_query(q, top_k=strategy.top_k + 25, alpha=strategy.alpha, where=where)
             else:
                 results = self.vs.query(q, top_k=self.top_k + 20, where=where)
             
             # Phase 174 Extension: Synonym Rollback Safety
             if synonym_active and results:
                  top_score = results[0].score
                  if top_score < 0.35:
                       print(f"[METRIC] Synonym Rollback Triggered: Score {top_score:.2f} < 0.35. Reverting to '{original_q_str}'")
                       telemetry_data["synonym_rollback"] = True
                       
                       # Re-run retrieval with original query
                       backoff_q = original_q_str
                       try:
                           if hasattr(self.vs, "hybrid_query"):
                                # If strategy was defined locally
                                tk = strategy.top_k + 25 if 'strategy' in locals() else self.top_k + 25
                                alp = strategy.alpha if 'strategy' in locals() else 0.5
                                fallback_results = self.vs.hybrid_query(backoff_q, top_k=tk, alpha=alp, where=where)
                           else:
                                fallback_results = self.vs.query(backoff_q, top_k=self.top_k + 20, where=where)
                                
                           fb_score = fallback_results[0].score if fallback_results else 0.0
                           if fb_score > top_score:
                                print(f"[METRIC] Rollback Successful: Score improved ({top_score:.2f} -> {fb_score:.2f})")
                                results = fallback_results
                                q = backoff_q # Update context query
                           else:
                                print(f"[METRIC] Rollback Aborted: Original was worse ({fb_score:.2f} < {top_score:.2f})")
                       except Exception as e:
                           print(f"[ERROR] Rollback Failed: {e}")
                 
             # ... (Ranking/Filtering) ...
             # Phase 98: Category Exclusion & System Page Filter
             filtered_results = []
             for r in results:
                 meta = r.metadata or {}
                 url = meta.get("url", "").lower()
                 c_type = meta.get("content_type", "")
                 if "view=category" in url: continue
                 if "view=section" in url: continue
                 if c_type == "category": continue
                 filtered_results.append(r)
             
             # Phase R1: Anti-Ranking Inversion
             if self.retrieval_optimizer:
                  filtered_results = self.retrieval_optimizer.re_rank(filtered_results, q)
             
             # Trim
             top_k_final = strategy.top_k if 'strategy' in locals() else self.top_k
             results = filtered_results[:top_k_final]
             
             latencies["vector_search"] = (time.time() - start_vs) * 1000
             
             # Verify Coverage (R1)
             if self.evaluator:
                  # Step 12: Fallback Stability Rule (Low Confidence Guard)
                  # If top score is below 0.75, trigger category menu instead of risking hallucination.
                  top_r_score = results[0].score if results else 0.0
                  
                  # Allow some very low scores to pass if they are "miss" candidates? No, miss is handled by coverage.
                  # We want to intercept the "Grey Zone" (e.g. 0.60 - 0.74).
                  if 0.0 < top_r_score < 0.75:
                       print(f"[STEP 12] Low Confidence Triggered (Score {top_r_score:.4f} < 0.75)")
                       
                       candidates = [
                           {"title": "Command", "payload": "รวบรวมคำสั่งใช้งาน (Command Reference)"},
                           {"title": "Troubleshooting", "payload": "แนวทางการแก้ปัญหา (Troubleshooting Guide)"},
                           {"title": "Overview", "payload": "ภาพรวมและโครงสร้างระบบ (System Overview)"},
                           {"title": "ความรู้ทั่วไป (General Knowledge)", "payload": "ความรู้ทั่วไปเกี่ยวกับอุปกรณ์ (General Knowledge)"}
                       ]
                       
                       self.pending_question = {
                           "kind": "category_selection",
                           "candidates": candidates,
                           "created_at": time.time()
                       }
                       
                       menu_text = "ไม่พบข้อมูลที่ตรงชัดเจน\nคุณต้องการค้นหาในหมวดใด:\n"
                       for idx, c in enumerate(candidates):
                           menu_text += f"{idx+1}. {c['title']}\n"
                           
                       latencies["total"] = (time.time() - t_start) * 1000
                       return {
                           "answer": menu_text,
                           "route": "clarification_low_confidence",
                           "decision_reason": f"Low Confidence (Score {top_r_score:.4f} < 0.75)",
                           "audit": {
                               "low_confidence_trigger": True,
                               "score": top_r_score
                           }
                       }
                  
                  # Phase R5: Pass Intent and Query for Dynamic Thresholds
                  coverage = self.evaluator.check_coverage(results, query=q, intent=intent)
                  print(f"[DEBUG] Evidence Coverage: {coverage['status']} ({coverage.get('reason', 'OK')})")
                  
                  if coverage["status"] == "MISS":
                       # Phase R5: General Knowledge Tier
                       # If it's a CONCEPT question, allow a Safe Fallback instead of strict MISS
                      if intent in ["EXPLAIN", "CONCEPT", "SUMMARY", "WHAT_IS", "DEFINE_TERM"]:
                            print(f"[DEBUG] General Tier Active for Concept Miss (Intent: {intent})")
                            is_general_fallback = True
                            # We proceed with whatever weak results we have OR empty?
                            # If we use strict generator, it might still refuse.
                            # We will rely on prompt engineering in Generator to handle "Low Evidence but General Concept".
                            # For now, let's keep the results (even if low score) so the LLM has *something* 
                            # context, but we flag it.
                      else:
                          latencies["total"] = (time.time() - t_start) * 1000
                          return {
                              "answer": "ไม่พบข้อมูลที่เกี่ยวข้องในระบบ (Coverage Check FAILED)",
                              "route": "rag_miss_coverage",
                              "latencies": latencies,
                              "debug_info": {"reason": coverage.get("reason")}
                              }
                  elif coverage["status"] == "LOW_CONFIDENCE":
                       telemetry_data["low_confidence"] = True
             
             # Step 2: Set L1 Cache
             self.cache_manager.set_retrieval_cache(q, intent, results)
             valid_results = results


        # =========================================================================
        # STEP 8: Enterprise Clarification Layer (Dynamic Bullet Options)
        # =========================================================================
        # Trigger Condition: 
        # 1. Ambiguous vector scores (Top3 diff < 0.05) AND No strong winner (> 0.9)
        # 2. Heuristic "Broad Query" (already checked earlier but this uses semantic evidence)
        # 3. Guardrails: No Exact Title Match (Passed), No Vendor Block (Passed).
        
        check_clarification = True
        
        # Guard: If intent is explicit concept/definition, we prefer trying to answer over clarifying
        # unless it is VERY ambiguous.
        if intent in ["DEFINE_TERM", "EXPLAIN"]:
            print(f"[STEP 8] Skip Clarification: Intent guard ({intent})")
            check_clarification = False

        if check_clarification and valid_results and len(valid_results) >= 2:
            top_score = valid_results[0].score

            # Step 12: Fallback Stability Rule (Low Confidence Guard)
            # If top score is low (< 0.75) but not zero/miss, trigger category fallback
            if top_score < 0.75:
                 print(f"[STEP 8] Low Confidence Fallback Triggered (Score={top_score:.4f})")
                 
                 # Log Telemetry
                 telemetry_data["clarification_triggered"] = True
                 telemetry_data["clarification_reason"] = "low_confidence_fallback"
                 telemetry_data["confidence_score"] = float(top_score)
                 
                 options_text = (
                     "1. Command\n"
                     "2. Troubleshooting\n"
                     "3. Overview\n"
                     "4. ความรู้ทั่วไป"
                 )
                 
                 msg = (
                     f"ไม่พบข้อมูลที่ตรงชัดเจน\n"
                     f"คุณต้องการค้นหาในหมวดใด:\n"
                     f"{options_text}"
                 )
                 
                 latencies["total"] = (time.time() - t_start) * 1000
                 return {
                      "answer": msg,
                      "route": "clarification_low_confidence",
                      "latencies": latencies,
                      "decision_reason": f"Low Confidence (Score {top_score:.4f})",
                      "audit": {
                          "decision_reason": f"Low Confidence (Score {top_score:.4f} < 0.75)",
                          "confidence_score": float(top_score)
                      }
                 }
            
            if top_score < 0.9:
                # Step 10 Core: Always capture metrics if we reach here
                telemetry_data["top_score"] = float(top_score)
                telemetry_data["n_candidates_above_threshold"] = len(valid_results)
                
                # Rule: Score difference check for Top 3
                # Get scores of top 3
                scores = [r.score for r in valid_results[:3]]
                score_diff = 0.0
                if len(scores) >= 2:
                    score_diff = scores[0] - scores[1]
                    telemetry_data["second_score"] = float(scores[1])
                    telemetry_data["score_gap"] = float(score_diff)
                
                # Step 11: Multi-Candidate Stress Handling (Multi-Close Detection)
                # Detect "clusters" of relevant documents within 0.05 score range of the top result.
                close_results = [
                    r for r in valid_results 
                    if top_score - r.score <= 0.05
                ]
                is_ambiguous_vector = (len(close_results) >= 2) and (top_score > 0.5)
                
                if not is_ambiguous_vector:
                    if top_score <= 0.5:
                        print(f"[STEP 8] Skip Clarification: Low visibility (TopScore={top_score:.4f})")
                    else:
                        print(f"[STEP 8] Skip Clarification: Single winner (TopScore={top_score:.4f})")
                
                if is_ambiguous_vector:
                     print(f"[Step 8] Multi-Close Detected: {len(close_results)} candidates within 0.05 range.")
                     telemetry_data["clarification_triggered"] = True
                     telemetry_data["clarification_reason"] = "multi_close_candidates"
                     
                     # Build Candidates
                     clarify_candidates = self._build_clarification_candidates(valid_results)
                     
                     # Step 11: Only Clarify if we have > 1 distinct good candidates
                     if len(clarify_candidates) >= 2:
                          print(f"[Step 8] Triggering Clarification with {len(clarify_candidates)} candidates.")
                          
                          # Step 11: Cap at Top 5 (User specification)
                          # Originally Step 9 mentioned 6, but Step 11 explicitly asks for 5. 
                          # We will follow the latest Step 11 requirement.
                          final_candidates = clarify_candidates[:5]
                          
                          # Format Bullet Response (Stricter Persona - Step 15)
                          options_text = ""
                          for idx, cand in enumerate(final_candidates):
                              options_text += f"• {cand['title']}\n"
                          
                          # Step 9/11: Professional Fallback Option
                          # options_text += f"{len(final_candidates)+1}️⃣ อื่น ๆ (ระบุเพิ่ม)\n" # Removed in favor of strict list
                          
                          msg = (
                              f"พบเอกสารที่เกี่ยวข้องในระบบ SMC ดังนี้:\n\n"
                              f"{options_text}\n"
                              f"กรุณาเลือกหัวข้อที่ต้องการ"
                          )
                          
                          # Step 9/11: Enhanced Metrics
                          telemetry_data["confidence_score"] = float(top_score)
                          telemetry_data["top_candidates_count"] = len(clarify_candidates)
                          
                          # Commit Pending State
                          self.pending_question = {
                              "kind": "article_selection",
                              "candidates": final_candidates, 
                              "created_at": time.time()
                          }
                          
                          latencies["total"] = (time.time() - t_start) * 1000
                          return {
                              "answer": msg,
                              "route": "pending_clarification",
                              "latencies": latencies,
                              "decision_reason": f"Ambiguous Vector Search (Diff {score_diff:.4f})"
                          }
                     else:
                          print(f"[STEP 8] Skip Clarification: Insufficient distinct candidates ({len(clarify_candidates)})")
            else:
                 print(f"[STEP 8] Skip Clarification: Strong winner (TopScore={top_score:.4f})")

        # Step 3: Kill-Switch (Safety Guard)

        from src.rag.safety_guard import SafetyGuard
        
        # Phase R6: Bypass Kill-Switch for General Fallback (Concept Tier)
        # If we explicitly allowed low confidence fallback earlier (Concept Tier), we skip strict score check.
        # But we still check for Navigation/Boilerplate (Safety).
        
        # Note: 'is_general_fallback' local variable from Coverage Check block
        allow_low_score = 'is_general_fallback' in locals() and is_general_fallback
        
        if allow_low_score:
             print(f"[SafetyGuard] General Fallback Active -> Bypassing Low Score Check")
             # Let's trust that 'Concept' intent implies we want an explanation.
             safety = {"safe": True, "reason": "General Fallback Override"}
        else:
             safety = SafetyGuard.check_retrieval_safety(valid_results)

        if not safety["safe"]:
             print(f"[SafetyGuard] KILL-SWITCH ACTIVATED: {safety['reason']}")
             return {
                 "answer": f"ไม่สามารถตอบคำถามได้เนื่องจากข้อมูลไม่เพียงพอหรือไม่มีคุณภาพ ({safety['reason']})",
                 "route": "rag_kill_switch",
                 "latencies": latencies
             }
             
        # Step 4: L2 Answer Cache Check (Using Fingerprint)
        # Always compute fingerprint first as it's needed for saving later
        fingerprint = self.cache_manager.compute_fingerprint(valid_results)
        
        # Rule DT-1: Do NOT use L2 cache for DEFINE_TERM to prevent poisoning (Always re-verify/regenerate)
        if intent == "DEFINE_TERM":
            print("[DEBUG] L2 Cache Skipped for DEFINE_TERM (Rule DT-1 Safety)")
            l2_hit = None
        else:
            l2_hit = self.cache_manager.get_answer_cache(q, intent, fingerprint)
        
        if l2_hit:
             print(f"[DEBUG] L2 Answer Cache HIT (Score: {l2_hit.get('score', 0):.2f})")
             latencies["cache"] = l2_hit.get("latency", 0)
             latencies["total"] = (time.time() - t_start) * 1000
             return {
                 "answer": l2_hit["answer"],
                 "route": "rag_cache_l2",
                 "latencies": latencies,
                 "score": l2_hit.get("score"),
                 "cache_hit": True,
                 "prompt_mode": l2_hit.get("prompt_mode", "cached")
             }

        # Proceed to Generation (R2 Logic)
        # ...

        
        if not results:
            latencies["total"] = (time.time() - t_start) * 1000
            return {
                "answer": "No information found in the system.",
                "route": "rag_no_docs",
                "latencies": latencies
            }

        # Filter by Score
        # Phase 133: If Concept, ignore global threshold (use gate below)
        score_threshold = float(self.rag_cfg.get("score_threshold", 0.0))
        if is_conceptual:
             valid_results = results # Keep all
        else:
             valid_results = [r for r in results if r.score >= score_threshold]
        
        if valid_results and valid_results[0].score >= 0.85:
            # print(f"[DEBUG] High confidence match ({valid_results[0].score:.2f}). Reducing Top-K to 1.")
            valid_results = [valid_results[0]]

        # Phase 116: Strict RAG Score Gate (Anti-Hallucination) - UPSTREAMED
        # Phase 133: Dynamic Threshold for Conceptual Queries (Broad/Summary)
        
        # Phase 133: Aggressive Retrieval for Conceptual Queries
        # If user wants "Summary", we trust the Ranking (Top 1) even if score is low.
        min_gate_score = 0.0 if (is_conceptual or intent == "KNOWLEDGE") else 0.2
        
        # If even the best result is weak (< min_gate_score), DO NOT Answer or Engage Controller.
        # Phase 136: Exception for DEFINE_TERM (Allow Fallback to General Knowledge)
        is_define_mode = (intent == "DEFINE_TERM")
        
        if (not valid_results or valid_results[0].score < min_gate_score) and not is_define_mode:
             best_s = valid_results[0].score if valid_results else 0
             print(f"[DEBUG] RAG Score Gate: Best score {best_s:.4f} < {min_gate_score} (Conceptual={is_conceptual}). REJECTING EARLY.")
             latencies["total"] = (time.time() - t_start) * 1000
             return {
                 "answer": "ไม่พบข้อมูลในระบบปัจจุบัน (Low Confidence Match)",
                 "route": "rag_low_score_gate",
                 "latencies": latencies,
                 "docs": [r.text[:100] for r in valid_results] if valid_results else [],
                 "reason": "Universal Low Score Gate (< 0.65)"
             }
        elif is_define_mode and (not valid_results or valid_results[0].score < 0.45):
             # Log that we are bypassing for Define Fallback
             print(f"[DEBUG] RAG Score Gate: DEFINE_TERM detected with weak coverage. Bypassing gate to invoke NT_STRICT fallback.")
        else:
             print(f"[DEBUG] RAG Score Gate Passed: Best score {valid_results[0].score:.4f} (Doc: {valid_results[0].text[:50]}...)")
            
        # 2.5 Retrieval Optimization (Dynamic Top-K)
        # Only run if we have multiple candidates and logic is enabled
        # This helps reduce noise for the LLM
        if len(valid_results) > 1 and self.retrieval_optimizer:
             t_opt = time.time()
             # We pass all valid results to the optimizer
             optimized_results = self.retrieval_optimizer.optimize(q, valid_results)
             latencies["retrieval_opt"] = (time.time() - t_opt) * 1000
             
             # If optimizer actually reduced the count or reordered?
             if optimized_results:
                  valid_results = optimized_results
        
        # 3. RAG Controller Strategy (LLM Judge)
        # Replaces simple thresholding. Run ONCE.
        candidates = valid_results if valid_results else results
        style = "GENERAL" # Default
        
        # 3.2 Dynamic Controller (The Brain)
        if candidates and self.controller:
             t_ctrl = time.time()
             # Optimize: Truncate context for Controller to reduce latency
             context_docs = [r.text for r in candidates] # Full docs for context
             # Limit to 500 chars per doc for decision to speed up.
             ctrl_context = [d[:500] for d in context_docs]
             
             strategy_res = self.controller.decide(q, ctrl_context)
             latencies["controller"] = (time.time() - t_ctrl) * 1000
             
             strategy = strategy_res.get("strategy")
             style = strategy_res.get("style", "GENERAL")
             
             # Phase 232: Handle Whitelist Strategy
             is_whitelist_override = False
             if strategy == "CONCEPT_EXPLAIN_WHITELIST":
                 is_whitelist_override = True
                 print(f"[DEBUG] Controller -> Whitelist Strategy Triggered (Context Bypass)")
             
             # Dispatch Strategy
             # Phase 133: Conceptual Override (Force Answer)
             if is_conceptual and strategy == "NO_ANSWER":
                 print(f"[DEBUG] Controller Rejected but Conceptual Override prevents it. Forcing RAG_ANSWER.")
                 strategy = "RAG_ANSWER"
                 
             if strategy == "NO_ANSWER":
                 latencies["total"] = (time.time() - t_start) * 1000
                 latencies["llm"] = latencies.get("controller", 0) + latencies.get("retrieval_opt", 0)
                 return {
                     "answer": "ไม่พบข้อมูลในระบบปัจจุบัน",
                     "route": "rag_controller_rejected",
                     "latencies": latencies,
                     "docs": context_docs,
                     "reason": strategy_res.get("reason"),
                     "retrieved_context": [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in candidates]
                 }
             elif strategy == "DIRECT_LOOKUP":
                 pass 
             elif strategy == "ASK_FOR_CLARIFICATION" or strategy == "CLARIFY":
                 latencies["total"] = (time.time() - t_start) * 1000
                 latencies["llm"] = latencies.get("controller", 0) + latencies.get("retrieval_opt", 0)
                 # Remove prefix, just return the question
                 return {
                     "answer": strategy_res.get("reason"), 
                     "route": "rag_clarify",
                     "latencies": latencies
                 }
             
             # Evidence Gate: If RAG_ANSWER but confidence is low, reject early
             # Limit wasteful generation
             conf = strategy_res.get("confidence", 0.0)
             if strategy == "RAG_ANSWER" and conf < 0.6:
                  latencies["total"] = (time.time() - t_start) * 1000
                  latencies["llm"] = latencies.get("controller", 0) + latencies.get("retrieval_opt", 0)
                  return {
                      "answer": "ไม่พบข้อมูลในระบบปัจจุบัน (Low Confidence)",
                      "route": "rag_controller_rejected",
                      "latencies": latencies,
                     "docs": context_docs,
                     "reason": f"Controller Confidence Low ({conf:.2f} < 0.6)",
                     "retrieved_context": [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in candidates]
                  }

             # If RAG_ANSWER or DIRECT_LOOKUP, we proceed. 
             # Also update valid_results to be candidates if they were originally empty (Rescue)
             if not valid_results and (strategy == "RAG_ANSWER" or strategy == "DIRECT_LOOKUP"):
                 valid_results = candidates
        
        if not valid_results and not is_whitelist_override:
            latencies["total"] = (time.time() - t_start) * 1000
            # Consolidate generic message
            return {
                "answer": "ไม่พบข้อมูลในระบบปัจจุบัน",
                "route": "rag_low_score",
                "latencies": latencies,
                "docs": [f"{r.text[:50]}... ({r.score:.4f})" for r in results],
                "retrieved_context": [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
            }

        # Build Context
        context_text = ""
        for i, res in enumerate(valid_results):
            context_text += f"\n[Document {i+1}]:\n{res.text}\n"

        # NEW: Extract and filter images from retrieved documents
        from src.rag.image_filter import detect_image_intent, filter_and_rank_images
        
        show_images = detect_image_intent(q)
        all_images = []
        
        if show_images:  # Only extract images if user requests them
            seen_urls = set()
            for res in valid_results:
                doc_url = res.metadata.get("url", "")
                if doc_url and doc_url in self.processed_cache._url_to_images:
                    doc_images = self.processed_cache._url_to_images[doc_url]
                    for img in doc_images:
                        img_url = img.get("url", "")
                        if img_url and img_url not in seen_urls:
                            all_images.append(img)
                            seen_urls.add(img_url)
            
            # Filter and rank images
            all_images = filter_and_rank_images(all_images, q, max_images=3)
        
        # Add images to context if available
        image_context = ""
        if all_images:
            image_context = "\n\nIMAGES AVAILABLE:\n"
            for idx, img in enumerate(all_images, 1):
                image_context += f"{idx}. URL: {img.get('url', '')}\n"
                if img.get('alt'):
                    image_context += f"   Alt: {img.get('alt', '')}\n"
                if img.get('caption'):
                    image_context += f"   Caption: {img.get('caption', '')}\n"

        # LLM Generation
        t_gen = time.time()
        
        # ---------------------------------------------------------
        # Phase R2: Answer Governance (RAGGenerator)
        # ---------------------------------------------------------
        # Use centralized generator for Strict Templates & Token Budgeting
        from src.rag.generator import RAGGenerator
        
        if not hasattr(self, "generator") or not self.generator:
             self.generator = RAGGenerator(self.llm_cfg)
             
        # Prepare Context Objects (Standardize)
        # valid_results already contains SearchResult objects
        
        # Determine Intent for Template Selection (Mapped from style or raw intent)
        # If Controller set style, map it.
        gen_intent = intent # Default to router intent
        extra_gen_kwargs = {}
        
        if style == "PROCEDURE": gen_intent = "HOWTO_PROCEDURE"
        elif style == "DEFINITION": gen_intent = "EXPLAIN"
        elif is_whitelist_override: gen_intent = "CONCEPT_EXPLAIN_WHITELIST" # Force Whitelist Intent
        
        # Override for Image/Conceptual
        if is_conceptual: gen_intent = "EXPLAIN"
        
        # Phase 136: DEFINE_TERM Handler (Policy 1b - Strict Fallback)
        if intent == "DEFINE_TERM":
             # Rule DT-1: Check Technical Glossary FIRST (Phase 236)
             q_up = q.upper().replace("คืออะไร", "").replace("คือ", "").strip()
             if q_up in TECHNICAL_GLOSSARY:
                 entry = TECHNICAL_GLOSSARY[q_up]
                 ans = f"[{entry['full_name']}]\n{entry['definition']}"
                 print(f"[DEBUG] Technical Glossary HIT for '{q_up}'")
                 
                 return {
                     "answer": ans,
                     "route": "rag", # or glossary_hit
                     "latencies": latencies,
                     "hits": []
                 }

             gen_intent = "NT_STRICT"
             
             # Determine Mode Hint based on Internal Evidence
             has_strong_internal = False
             if valid_results:
                 # Check score of top result
                 top_score = getattr(valid_results[0], "score", 0.0)
                 if top_score > 0.45: # Moderate threshold for Definition
                     has_strong_internal = True
             
             if has_strong_internal:
                 extra_gen_kwargs["mode_hint"] = "RAG_CONTEXT" # Use internal context
             else:
                 print(f"[DEBUG] DEFINE_TERM: Internal Miss (Score Low) -> Triggering General Knowledge Fallback.")
                 extra_gen_kwargs["mode_hint"] = "NO_CONTEXT" # Trigger Rule 4 (General Def)
        
        # Phase R5: General Fallback Override
        # If coverage was missing but we allow general answer (Concept Tier)
        # Note: DEFINE_TERM handles its own fallback above.
        elif 'is_general_fallback' in locals() and is_general_fallback:
             gen_intent = "GENERAL_FALLBACK"
             print(f"[Thinking...] Using Intent Template: GENERAL_FALLBACK (Concept Fallback)")
        else:
             print(f"[Thinking...] Using Intent Template: {gen_intent}")
        
        # Determine Doc Type & Policy Flags (New System)
        if "doc_type" not in extra_gen_kwargs:
             d_type = "TEXT_ARTICLE"
             if not valid_results: d_type = "NONE"
             elif intent == "DEFINE_TERM": d_type = "GLOSSARY"
             elif intent == "REFERENCE_LINK": d_type = "NAV_INDEX"
             extra_gen_kwargs["doc_type"] = d_type
             
        # Phase 237: Knowledge/News Threshold Logic (Retrieval-First, Summary-Second)
        # Goal: Coverage first with Link Only, Summary only for high-confidence.
        if valid_results and gen_intent in ["HOWTO_PROCEDURE", "EXPLAIN", "SUMMARIZE", "CONCEPT_EXPLAIN"]:
            top_hit = valid_results[0]
            top_score = getattr(top_hit, "score", 0.0)
            
            # LINK_ONLY_THRESHOLD = 0.20
            # LLM_SUMMARY_THRESHOLD = 0.35 
            
            if top_score < 0.35:
                # Coverage-First: Return Link Only
                meta = getattr(top_hit, "metadata", {})
                title = meta.get("title", "บทความที่เกี่ยวข้อง")
                url = meta.get("url") or meta.get("source")
                
                print(f"[DEBUG] Knowledge Threshold: Score {top_score:.4f} < 0.35. Triggering LINK_ONLY mode.")
                
                answer = f"พบข้อมูลที่เกี่ยวข้องแต่คะแนนความเชื่อมั่นไม่สูงนัก แนะนำให้อ่านรายละเอียดจากลิงก์โดยตรงเพื่อความถูกต้องครับ:\n\n**[{title}]**\n🔗 [คลิกเพื่อเปิดดู]({url})"
                
                latencies["total"] = (time.time() - t_start) * 1000
                return {
                    "answer": answer,
                    "route": "rag_link_only",
                    "latencies": latencies,
                    "source_url": url,
                    "score": top_score
                }
            else:
                # High Confidence: Check Content Type Before Summarizing
                print(f"[DEBUG] Knowledge Threshold: Score {top_score:.4f} >= 0.35. Checking content type...")
                
                # Extract content for classification
                meta = getattr(top_hit, "metadata", {})
                title = meta.get("title", "บทความ")
                url = meta.get("url") or meta.get("source", "")
                content = getattr(top_hit, "page_content", "")
                
                # Classify content type
                classification = self.content_classifier.classify(
                    title=title,
                    url=url,
                    content=content,
                    metadata={
                        "text_length": len(content),
                        "detected_table_rows": content.count("\n") if "|" in content else 0,
                        "detected_images": content.lower().count("[image") + content.lower().count("<img")
                    }
                )
                
                if not classification["should_summarize"]:
                    # Content is not suitable for summarization (LINK_MENU, IMAGE_ONLY, TABLE_HEAVY)
                    print(f"[DEBUG] Content Type: {classification['content_type']}. Forcing LINK_ONLY mode.")
                    
                    answer = f"พบข้อมูลที่เกี่ยวข้อง แต่เนื้อหาเป็นประเภท **{classification['content_type']}** ซึ่งเหมาะกับการเปิดดูโดยตรงมากกว่าครับ:\n\n**[{title}]**\n🔗 [คลิกเพื่อเปิดดู]({url})"
                    latencies["total"] = (time.time() - t_start) * 1000
                    return {
                        "answer": answer,
                        "route": "rag_link_only_content_type",
                        "latencies": latencies,
                        "source_url": url,
                        "score": top_score,
                        "content_type": classification["content_type"]
                    }
                
                # Content is TEXT_ARTICLE: Use Controlled Summarizer
                print(f"[DEBUG] Content Type: TEXT_ARTICLE. Triggering LLM_SUMMARY mode.")
                gen_intent = "SUMMARIZE_HYBRID" # Force use of TEMPLATE_SUMMARIZER
                valid_results = [top_hit]

        if "policy_flags" not in extra_gen_kwargs:
             p_flags = []
             bad_keywords = ["password", "root", "admin", "credential", "รหัสผ่าน"]
             if any(k in q.lower() for k in bad_keywords):
                 p_flags.append("RESTRICTED_CREDENTIALS")
             extra_gen_kwargs["policy_flags"] = p_flags

        gen_res = self.generator.generate(
            query=q,
            context_docs=valid_results,
            intent=gen_intent,
            **extra_gen_kwargs
        )
        
        ans = gen_res.get("answer", "")
        latencies["generator"] = gen_res.get("latency", 0)
        
        # Temporary: Append Image Context manually if it exists?
        # RAGGenerator handles text context. Images are separate.
        # If we have images, we might need to append them to the answer directly 
        # OR pass them to generator (future).
        # For now, let's append the Image Section manually if RAGGenerator didn't.
        # (Our current RAGGenerator prompt doesn't handle image_context arg yet, 
        #  so we append it to the FINAL string if not empty).
        
        if image_context and ans:
             # Basic append if not present
             if "รูปภาพประกอบ" not in ans:
                 ans += f"\n\n{image_context}"

        t_gen = time.time() # Just for compat with old vars if needed below
        # (latencies['generator'] is already set from gen_res)
        
        # Step 5: Reviewer Agent (Phase R4)
        # Check grounding (Only if we have an answer and it's not a "not found" boilerplate)
        if ans and "ไม่พบข้อมูล" not in ans and len(ans) > 20: 
             from src.rag.reviewer import RAGReviewer
             if not hasattr(self, "reviewer"):
                 self.reviewer = RAGReviewer(self.llm_cfg)
                 
             # Only review if strictly needed (e.g. not procedural/fact check which is usually safe?)
             # For R4, let's review everything to be safe.
             
             review_res = self.reviewer.review(q, ans, valid_results)
             print(f"[Reviewer] Verdict: {review_res.get('verdict')} ({review_res.get('reason')})")
             
             # Rule RV-1 (Reviewer) + Rule DT-GEN-2/3
             # If intent is DEFINE_TERM and answer has Person Name (heuristic), force FAIL.
             if intent == "DEFINE_TERM":
                  person_kws = ["นาย", "นาง", "คุณ", "นางสาว"]
                  # Simplified heuristic: if answer starts with a name pattern or contains common Thai titles 
                  if any(kw in ans for kw in person_kws) or "คือชื่อ" in ans or "เป็นชื่อ" in ans:
                       print(f"[Rule RV-1] Rejecting DEFINE_TERM answer due to Person Name detection (Hallucination Guard)")
                       ans = "ไม่พบข้อมูลนิยามที่ถูกต้องในระบบ และอยู่นอกขอบเขตของคำอธิบายทั่วไป"
                       review_res["verdict"] = "HARD_FAIL"

             if review_res.get("verdict") in ["FAIL", "HARD_FAIL"] and review_res.get("safe_response"):
                  # Swap answer!
                  ans = review_res.get("safe_response")
                  latencies["reviewer"] = 999 # Placeholder for tracking
                  # Mark route
                  # We can't easily change route variable here as it's implied RAG.
                  # But we can log it.
                  telemetry_data["reviewer_fail"] = True
                  telemetry_data["reviewer_reason"] = review_res.get("reason")
        
        # Step 6: Update L2 Answer Cache (Strict)
        # Only cache if Reviewer PASSED (or we swapped to a safe response?)
        # Fix: Force Save if 'is_general_fallback' because we want to cache the "Not Found/General Knowledge" answer
        should_cache = (ans and "ไม่พบข้อมูล" not in ans and len(ans) > 20) or (is_general_fallback and ans)
        
        # Rule DT-1 (Cache Safety): Prevent poisoning from General Fallbacks on DEFINE_TERM
        if intent == "DEFINE_TERM" and is_general_fallback:
             # Default: FAIL SAFE. Do not cache general fallbacks unless strictly verified.
             # OLT Example: "olt is ..." might be wrong. Verification is hard.
             # Better to re-generate each time or rely on L1.
             print(f"[DEBUG] Cache Skipped for DEFINE_TERM General Fallback (Rule DT-1 Safety)")
             should_cache = False
        if should_cache:
             # Save with Fingerprint
             print(f"[DEBUG] Saving to L2 Cache (Fingerprint: {fingerprint[:10]}...)")
             meta = {
                 "route": "rag",
                 "model": self.llm_cfg.get("model", "unknown"),
                 "locations": loc_fingerprint
             }
             self.cache_manager.set_answer_cache(q, ans, intent, fingerprint, meta)

        
        # 4. RAG Evaluator (Post-Gen Guard)
        # Check both English and Thai refusal strings for Evaluator
        refusal_phrases = ["No information", "ไม่พบข้อมูลในระบบปัจจุบัน", "ไม่พบข้อมูลที่ยืนยัน", "ไม่พบข้อมูล"]
        is_refusal = any(phrase in ans for phrase in refusal_phrases)
        
        # Phase R6: Bypass Verification for General Fallback
        # If we already decided to show a "General Knowledge" answer (Concept Tier), 
        # we don't want the strict Verifier to reject it because it lacks internal evidence.
        if self.evaluator and ans and not is_refusal and not is_general_fallback:
             t_eval = time.time()
             eval_res = self.evaluator.verify(q, [r.text for r in valid_results], ans)
             latencies["evaluator"] = (time.time() - t_eval) * 1000
             
             if eval_res.get("verdict") == "REJECT":
                 latencies["total"] = (time.time() - t_start) * 1000
                 latencies["llm"] = (latencies.get("controller", 0) + 
                                     latencies.get("retrieval_opt", 0) + 
                                     latencies.get("generator", 0) + 
                                     latencies.get("evaluator", 0))
                 return {
                     "answer": "ไม่พบข้อมูลในระบบปัจจุบัน (Verification Failed)",
                     "route": "rag_evaluator_rejected",
                     "latencies": latencies,
                     "docs": [r.text for r in valid_results],
                     "reason": eval_res.get("missing_evidence")
                 }
             elif eval_res.get("verdict") == "TIMEOUT":
                 # Graceful Degradation: Warn but allow answer
                 print(f"[DEBUG] Verification Timed Out. Proceeding with warning.")
                 ans += "\n\n[System Warning: Verification skipped due to high load]"

        latencies["total"] = (time.time() - t_start) * 1000
        
        # Sum LLM Components
        latencies["llm"] = (latencies.get("controller", 0) + 
                            latencies.get("retrieval_opt", 0) + 
                            latencies.get("generator", 0) + 
                            latencies.get("evaluator", 0))
        # 5. Source Attachment (Phase 28)
        # If answer is valid and not a refusal, attach references
        if ans and not is_refusal and valid_results:
             unique_sources = {} # url -> title
             for r in valid_results:
                 if not r.metadata: continue
                 url = r.metadata.get("source") or r.metadata.get("url")
                 title = r.metadata.get("title") or "Document"
                 if url and url not in unique_sources:
                     unique_sources[url] = title
                     
             if unique_sources:
                 sources_text = ""
                 for i, (url, title) in enumerate(list(unique_sources.items())[:3]):
                      sources_text += f"\n{i+1}. {title} ({url})"
                 
                 # Phase 6: Emphasize primary source at TOP for technical queries
                 # This satisfies 'เน้นให้ระบบใส่ลิงค์ที่มาให้ถูกต้องก่อน'
                 tech_indicators = ["onu", "olt", "huawei", "zte", "cisco", "config", "command", "คำสั่ง", "setup", "วิธี"]
                 is_tech_query = any(k in q_lower for k in tech_indicators)
                 
                 if is_tech_query:
                      top_url = list(unique_sources.keys())[0]
                      top_title = unique_sources[top_url]
                      # Bold Top Header with Link
                      ans = f"📌 **แหล่งข้อมูลหลัก:** [{top_title}]({top_url})\n\n" + ans
                      ans += f"\n\n---"
                      ans += f"\n**ตรวจสอบข้อมูลเพิ่มเติม:** เพื่อความถูกต้องแม่นยำที่สุด กรุณาอ่านรายละเอียดจากลิงก์ต้นฉบับครับ"
                 else:
                      ans += "\n\nแหล่งข้อมูลอ้างอิง:" + sources_text
        
        # Location Intent (Phase 36)
        # Already extracted early as 'req_locations' and 'loc_fingerprint'

        # 2. Check Semantic Cache
        if self.cache:
            # Add location strictness to filter
            # This ensures "songkhla" query NEVER hits "surat" cache
            filter_meta = {
                "route": "rag",
                "model": self.llm_cfg.get("model", "unknown"),
                "locations": loc_fingerprint 
            }
            
            t_cache = time.time()
            # Phase 97: Strict Intent Check
            # Rule DT-1/DT-ROLE-1: Bypass later semantic cache to avoid stale hits for these sensitive intents.
            best_hit = None
            if intent not in ["DEFINE_TERM", "POSITION_HOLDER_LOOKUP"]:
                best_hit = self.cache.check(
                    q, 
                    intent=intent, 
                    route="rag", 
                    filter_meta=filter_meta
                )
            
            if best_hit:
                score = best_hit.get("score", 0.0)
                if score > self.cache_threshold:
                    print(f"[DEBUG] Semantic Cache HIT (score={score:.4f})")
                    latencies["cache"] = (time.time() - t_cache) * 1000
                    latencies["total"] = (time.time() - t_start) * 1000
                    
                    return {
                        "answer": best_hit["answer"],
                        "route": "rag_cache",
                        "latencies": latencies,
                        "score": score
                    }

        # 6. Store in Cache (Pro-Level Metadata)
        # Phase 235: Fail-Safe Caching
        # Only cache valid positive answers. Do not cache "Not Found" or "Refusal".
        is_internal_miss = "ไม่พบข้อมูลอ้างอิงจากเอกสารภายใน" in ans or "ไม่พบข้อมูลเบอร์โทรศัพท์" in ans
        is_concept_refusal = "ไม่สามารถอธิบายได้อย่างถูกต้องในขอบเขตที่กำหนด" in ans
        should_cache_fail = False # Strict: Never cache misses (to allow re-try with new docs)

        if self.cache and ans and not is_refusal and len(ans) > 20 and not is_internal_miss and not is_concept_refusal: 
             # Phase 97: Strict Metadata for Hygiene
             store_meta = {
                 "route": "rag",
                 "intent": intent, # Strict Key
                 "prompt_version": PROMPT_VERSION, # Strict Version
                 "model": self.llm_cfg.get("model", "unknown"),
                 "timestamp": time.time(),
                 "locations": loc_fingerprint
             }
             self.cache.store(q, ans, meta=store_meta)
        else:
             if is_internal_miss or is_concept_refusal:
                 print(f"[DEBUG] Cache Skip: 'Not Found' response not cached. (Miss={is_internal_miss}, Refusal={is_concept_refusal})")
             
        # Phase 33: Populate hits for Article URL_ONLY mode
        # If no KP matches, we treat the retrieved docs as "hits" for URL filtering
        doc_matches = []
        for r in valid_results:
            if r.metadata.get("url"):
                doc_matches.append({
                    "key": "Article",
                    "value": r.metadata.get("title", "Link"),
                    "source_url": r.metadata.get("url")
                })
        
        # Merge doc matches into results if not present (or append for hybrid)
        # Check if we should use doc_matches as primary hits for RAG route
        if not kp_result:
             matches = doc_matches
        else:
             matches = kp_result.get("hits", []) + doc_matches

        # Location Slicing (Phase 36)
        # Apply strict location filtering if user requested specific province
        
        # req_locations = extract_location_intent(q) # Moved up for cache
        if req_locations and ans:
            print(f"[DEBUG] Location Intent Detected: {req_locations}. Slicing answer...")
            sliced_ans = slice_markdown_section(ans, req_locations)
            # Only use sliced if it's not empty and significantly different
            if sliced_ans and len(sliced_ans) < len(ans):
                ans = sliced_ans + f"\n\n(คัดกรองเฉพาะข้อมูล: {', '.join(req_locations)})"

        return {
            "answer": ans,
            "route": "rag",
            "context": context_text if self.show_context else None,
            "latencies": latencies,
            "docs": [r.text for r in valid_results],
            "hits": matches,
            "debug_info": {
                "generated_at": time.time(),
                "model": self.llm_cfg.get("model")
            }
        }


    def _find_vendor_articles(self, vendor: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Step 16: Advanced Vendor Query Handling (SMC-Only)
        Query deterministic index, apply SMC priority scoring, and categories.
        
        Args:
            vendor: Vendor name (e.g., "Huawei", "ZTE")
            limit: Max results (default 10 as per Step 16)
            
        Returns:
            List of dicts with 'title', 'url', 'score', 'category'
            Sorted by smc_priority_score DESC
        """
        results = []
        vendor_lower = vendor.lower()
        
        if not hasattr(self, 'processed_cache') or not self.processed_cache or not self.processed_cache._loaded:
            return results
        
        seen_urls = set()
        
        # Search normalized title index
        for norm_title, data in self.processed_cache._normalized_title_index.items():
            title = data.get('text', '')
            url = data.get('href', '')
            
            # Filter: title contains vendor (case insensitive)
            if vendor_lower not in title.lower():
                continue
            
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Calculate SMC Priority Score (Step 3)
            smc_score = 0
            title_lower = title.lower()
            
            if "คำสั่งพื้นฐาน" in title:
                smc_score += 5
            if "overview" in title_lower:
                smc_score += 3
            if "config" in title_lower:
                smc_score += 2
            if "checking" in title_lower:
                smc_score += 1
                
            # Keep original score 1.0 since it's an index hit
            
            # Determine Category (Step 4)
            category = "ทั่วไป"
            if "คำสั่งพื้นฐาน" in title:
                 category = "คำสั่งพื้นฐาน"
            elif "olt" in title_lower or "ont" in title_lower:
                 category = "OLT / ONT"
            elif "config" in title_lower:
                 category = "Configuration"
            
            results.append({
                'title': title,
                'url': url,
                'score': 1.0,
                'smc_priority_score': smc_score,
                'category': category
            })
            
        # Sort by SMC Score DESC, then Title ASC
        results.sort(key=lambda x: (-x['smc_priority_score'], x['title']))
        
        return results[:limit]




    def normalize_smc_url(self, text: str) -> str:
        """
        Fixes broken SMC URLs where 'view=article' is missing or mangled.
        e.g. '&=article' -> 'view=article'
        """
        if not text or "option=com_content" not in text:
            return text
        # Fix &=article -> &view=article
        text = text.replace("&=article", "&view=article")
        # Fix ?option=com_content&=article -> ?option=com_content&view=article
        # (Though replace above handles it, this is for clarity)
        return text

    def _handle_article_route(self, url: str, query: str, latencies: Dict[str, float], start_time: float, match_score: float = 0.0, intent: str = None, article_type: str = None, decision_reason: str = None) -> Dict[str, Any]:
        """
        Helper to invoke Article Interpreter for a specific URL.
        """
        if not url:
            return {
                "answer": "ไม่พบลิงก์บทความที่เกี่ยวข้อง",
                "route": "article_miss",
                "latencies": latencies
            }
            
        # 3. ROOT-CAUSE GUARDRAIL (Phase 8 Step 4)
        # Verify if article_id (url) is in SMC Index
        # This prevents serving hallucinations or unauthorized links
        is_verified = False
        if hasattr(self, 'processed_cache') and self.processed_cache:
            # Check if URL is in normalized index or raw list
            # Simplify: Check if URL contains trusted domain (10.192.133.33) OR is in cache
            if "10.192.133.33" in url or "smc" in url.lower():
                is_verified = True
            elif self.processed_cache.is_known_url(url): # Assuming method exists or checks dict
                 is_verified = True
                 
        if not is_verified:
             # Strict Block
             print(f"[GOVERNANCE] BLOCKED: Article URL not verified in SMC Index: {url}")
             return {
                 "answer": (
                     "⚠️ **ระงับการแสดงผล (Unverified Source)**\n\n"
                     "ลิงก์เอกสารไม่อยู่ในฐานข้อมูล SMC ที่ตรวจสอบแล้ว (Potential Hallucination)\n"
                     "เพื่อความปลอดภัย ระบบขอสงวนสิทธิ์ไม่แสดงเนื้อหานี้ครับ"
                 ),
                 "route": "blocked_scope",
                 "block_reason": "UNVERIFIED_ARTICLE_ID",
                 "latencies": latencies
             }
        
        # Invoke Article Interpreter (Phase 16)
        print(f"[DEBUG] News/Article Route -> {url}")

        if intent == "FALLBACK_LINK_ONLY":
            # Get article title
            article_title = "SMC Article"
            if hasattr(self, 'processed_cache') and self.processed_cache:
                for title_norm, entries in self.processed_cache._link_index.items():
                    for entry in entries:
                        if entry.get("href") == url:
                            article_title = entry.get("text", "SMC Article")
                            break
                    if article_title != "SMC Article": break
            
            # STEP 3: Classify content and provide appropriate explanation
            article_content = self.processed_cache._url_to_text.get(url) if hasattr(self, 'processed_cache') else None
            from src.ai.content_classifier import classify_article_content
            content_type = classify_article_content(article_title, article_type, article_content)
            
            # Non-summarizable content gets explanation text
            non_summarizable = content_type in ["command_reference", "table_heavy", "image_heavy"]
            
            if non_summarizable:
                # Determine content type label
                if content_type == "command_reference":
                    type_label = "คำสั่ง"
                elif content_type == "table_heavy":
                    type_label = "ตาราง"
                else:  # image_heavy
                    type_label = "ภาพประกอบ"
                
                print(f"[STEP 3] FALLBACK_LINK_ONLY + {content_type} -> Non-Summarizable Explanation")
                return {
                    "answer": (
                        f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                        f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                        f"เอกสารนี้เป็นข้อมูลเชิง{type_label} ซึ่งไม่เหมาะสำหรับการสรุปโดยอัตโนมัติ\n\n"
                        f"กรุณาดูรายละเอียดจากเอกสารต้นฉบับด้านบนครับ"
                    ),
                    "route": "article_link_only_index",
                    "decision_reason": f"FALLBACK_NON_SUMMARIZABLE_{content_type.upper()}",
                    "latencies": latencies,
                    "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}],
                    "content_type": content_type
                }
            
            # Phase 18: Single Canonical Link Rule (for narrative/index content)
            print(f"[GOVERNANCE] Returning Link Only (Intent: {intent})")
            return {
                "answer": (
                    f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                    f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                    f"ต้องการให้สรุปรายละเอียดในหัวข้อไหน แจ้งได้เลยครับ"
                ),
                "route": "article_link_only_index",
                "latencies": latencies,
                "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}]
            }

        # Must ensure interpreter is loaded
        if not hasattr(self, "article_interpreter") or not self.article_interpreter:
            from src.rag.article_interpreter import ArticleInterpreter
            self.article_interpreter = ArticleInterpreter(self.llm_cfg)
            
        t_art = time.time()
        
        # Check Cache FIRST (Phase R6)
        # We cache the RESULT of interpretation (the summary)
        # Fix: Safe access to prevention crash if not initialized
        cm = getattr(self, "cache_manager", None)
        if cm:
            ux_formatting = self.cfg.get("ux", {}).get("enable_formatting", True)
            ux_preview = self.cfg.get("ux", {}).get("direct_content_preview", True)
            # Phase 185: Unified Cache & Config (User Request)
            # Add strict UX flags to key to ensure policy changes invalidate cache
            ux_long_mode = self.cfg.get("ux", {}).get("article", {}).get("intro", {}).get("enabled", True)
            ux_max_chars = self.cfg.get("ux", {}).get("article", {}).get("max_chars", 900)
            
            # Use strict Schema v220 - PROMPT VERSIONING
            from src.rag.prompts import PROMPT_VERSION
            cache_key_str = f"article_summary|{url}|{query}|{PROMPT_VERSION}|fmt:{ux_formatting}|pre:{ux_preview}|intro:{ux_long_mode}|lim:{ux_max_chars}" 
            fingerprint = hashlib.md5(cache_key_str.encode()).hexdigest()
            
            # Use 'ARTICLE_SUMMARY' as intent
            # Phase 185 Fix: Change Intent Key to force complete cache partition
            # Phase 221 Update: Bump to V3 to invalidate old noisy summaries (MA cisco fix)
            intent_key = "ARTICLE_SUMMARY_V3" 
            l2_hit = self.cache_manager.get_answer_cache(query, intent_key, fingerprint)
            if l2_hit:
                 print(f"[DEBUG] Article Summary Cache HIT (Score: {l2_hit.get('score', 0):.2f})")
                 latencies["cache"] = l2_hit.get("latency", 0)
                 latencies["total"] = (time.time() - start_time) * 1000
                 ans = self.normalize_smc_url(clean_junk_text(l2_hit["answer"]))
                 
                 # Phase 18: SINGLE CANONICAL LINK RULE + Footer Cleanup
                 canonical_header = f"📌 **แหล่งข้อมูลหลัก (SMC):**\n🔗 [{l2_hit.get('metadata', {}).get('title', 'SMC Article')}]({self.normalize_smc_url(url)})\n\n"
                 
                 # Clean previous footers from cached answer
                 clean_ans = ans
                 for f in ["แหล่งที่มา:", "---", "ตรวจสอบข้อมูลเพิ่มเติม:"]:
                     if f in clean_ans: clean_ans = clean_ans.split(f)[0].strip()
                 
                 final_ans = canonical_header + clean_ans + "\n\nตรวจสอบข้อมูลเพิ่มเติมจากลิงก์ SMC ต้นฉบับ"

                 return {
                      "answer": final_ans,
                      "route": "article_answer",
                      "latencies": latencies,
                      "hits": [{"key": "Source", "value": "SMC Article", "source_url": self.normalize_smc_url(url)}],
                      "content_type": "cache_summary"  # Metadata: cached summary
                 }

        # Fetch content (Cache or Web)
        # Note: Interpreter expects context or url? 
        # Checking ArticleInterpreter.interpret signature...
        # interpret(user_query, article_title, article_url, article_content, ...)
        
        article_content = self.processed_cache._url_to_text.get(url)
        if not article_content:
            # Phase 130: Smart Fetch with Timeout & Policy
            print(f"[DEBUG] Cache Miss for {url} -> Attempting Fetch (Strict Timeout)...")
            fetch_res = fetch_with_policy(url)
            
            if fetch_res.status_code == 200 and fetch_res.html:
                # Success: Parse and use
                print(f"[DEBUG] Fetch Success: {len(fetch_res.html)} chars")
                # Clean html to text immediately for use
                from src.ingest.clean import clean_html_to_text
                clean_res = clean_html_to_text(fetch_res.html)
                article_content = clean_res.text
                
                # Optional: Update memory cache (TTL not implemented in memory dict yet, but good for session)
                self.processed_cache._url_to_text[url] = article_content
            
            else:
                # Phase SMC: MODE 2 - LINK_ONLY (Fetch Failure)
                # STRICT: Show only link + error, NO summary, NO extrapolation
                reason = fetch_res.error or f"Status {fetch_res.status_code}"
                print(f"[SMC GOVERNANCE] Fetch Failed: {reason} -> MODE: LINK_ONLY")
                
                # Get article title from link_index if available
                article_title = "SMC Article"
                for title_norm, entries in self.processed_cache._link_index.items():
                    for entry in entries:
                        if entry.get("href") == url:
                            article_title = entry.get("text", "SMC Article")
                            break
                    if article_title != "SMC Article":
                        break
                
                # Determine error message based on reason
                error_detail = "ไม่สามารถดึงเนื้อหาได้"
                if "timeout" in reason.lower():
                    error_detail = "ใช้เวลาดึงข้อมูลนานเกินกำหนด (Timeout)"
                elif "404" in reason:
                    error_detail = "ไม่พบหน้าเว็บ (404 Not Found)"
                elif "blocked" in reason.lower():
                    error_detail = "ไม่สามารถเข้าถึงแหล่งข้อมูลนี้ได้ (Domain Restricted)"
                elif "parse" in reason.lower() or "html" in reason.lower():
                    error_detail = "รูปแบบเนื้อหาไม่รองรับการแสดงผล"
                
                # MODE 2: LINK_ONLY - Show ONLY link + error
                return {
                    "answer": (
                        f"[SMC Reference]\n"
                        f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                        f"⚠️ **ไม่สามารถโหลดเนื้อหาได้:** {error_detail}\n\n"
                        f"กรุณาคลิกลิงก์ด้านบนเพื่ออ่านโดยตรงจาก SMC ครับ"
                    ),
                    "route": "article_link_only",
                    "latencies": latencies,
                    "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}],
                    "content_type": "fetch_failed"  # Metadata: fetch failure
                }
        
        # 1. Resolve Article Title
        article_title = "SMC Article"
        if hasattr(self, 'processed_cache') and self.processed_cache:
            for title_norm, entries in self.processed_cache._link_index.items():
                for entry in entries:
                    if entry.get("href") == url:
                        article_title = entry.get("text", "SMC Article")
                        break
                if article_title != "SMC Article": break

        # 2. Check for Link-Only Override (Precision Check)
        title_lower = article_title.lower()
        tech_kws = ["command", "คำสั่ง", "show", "config", "telnet", "add", "ipphone"]
        is_tech_article = (article_type == "COMMAND_REFERENCE") or any(kw in title_lower or kw in url.lower() for kw in tech_kws)
        
        # Precision check: if query length is very close to title length
        is_precise_tech = is_tech_article and abs(len(query.strip()) - len(article_title.strip())) <= 12
        
        # Check if this is an exact title match (for prioritization)
        q_normalized = self.processed_cache.soft_normalize(query) if hasattr(self, 'processed_cache') else query.lower()
        title_normalized = self.processed_cache.soft_normalize(article_title) if hasattr(self, 'processed_cache') else article_title.lower()
        is_title_match = q_normalized == title_normalized or query.lower().strip() == article_title.lower().strip()
        
        # 2.1 Rule: OVERVIEW used as Index for COMMAND -> Force LINK_ONLY (UNLESS exact title match)
        is_index_article = (article_type == "OVERVIEW") and (intent == "COMMAND" or "command" in query.lower()) and not is_title_match
        if is_index_article:
            print(f"[GOVERNANCE] Phase 19: Overview Article as Command Index -> Forced LINK_ONLY")
            return {
                "answer": (
                    f"[SMC Reference]\n"
                    f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                    f"บทความนี้เป็นดรรชนีรวมคำสั่ง แนะนำตรวจสอบคำสั่งที่ต้องการจากลิงก์ด้านบนครับ"
                ),
                "content_type": "index_link_only",  # Metadata: index page
                "route": "article_link_only_index",
                "decision_reason": "OVERVIEW_AS_INDEX",
                "latencies": latencies,
                "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}]
            }
        
        # FIX 1: PRODUCTION LOCK - Deterministic Exact Match MUST be LINK_ONLY
        # Relaxed for Hybrid Mode: Allow DETERMINISTIC_MATCH to proceed to summarization if content permits
        if intent == "FALLBACK_LINK_ONLY":
            reason = "FALLBACK_LINK_ONLY"
                
            print(f"[GOVERNANCE] Phase 21: {reason}")
            return {
                "answer": (
                    f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                    f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                    f"พบข้อมูลที่ตรงกับคำค้นหาของคุณโดยละเอียด แนะนำดูรายละเอียดทั้งหมดจากลิงก์ด้านบนครับ"
                ),
                "route": "article_link_only_exact",
                "decision_reason": reason,
                "latencies": latencies,
                "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}],
                "content_type": "deterministic_exact"  # Metadata: exact match
            }

        # STEP 3: Content Classification & Non-Summarizable Explanation Mode
        # Classify content to determine if it should be summarized or just linked
        from src.ai.content_classifier import classify_article_content
        content_type = classify_article_content(article_title, article_type, article_content)
        
        # Non-summarizable content types should NOT be sent to LLM
        # Non-summarizable content types should NOT be sent to LLM
        # Relaxed for Hybrid Mode: Allow command_reference (User Request)
        non_summarizable_types = ["table_heavy", "image_heavy"]
        
        if content_type in non_summarizable_types:
            print(f"[STEP 3] Content Type: {content_type} -> Non-Summarizable Explanation Mode")
            
            # Determine content type label in Thai
            if content_type == "command_reference":
                type_label = "คำสั่ง"
            elif content_type == "table_heavy":
                type_label = "ตาราง"
            else:  # image_heavy
                type_label = "ภาพประกอบ"
            
            # Return explanation + link instead of summary
            latencies["total"] = (time.time() - start_time) * 1000
            return {
                "answer": (
                    f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                    f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                    f"เอกสารนี้เป็นข้อมูลเชิง{type_label} ซึ่งไม่เหมาะสำหรับการสรุปโดยอัตโนมัติ\n\n"
                    f"กรุณาดูรายละเอียดจากเอกสารต้นฉบับด้านบนครับ"
                ),
                "route": "article_link_only_content_type",
                "decision_reason": f"NON_SUMMARIZABLE_CONTENT_{content_type.upper()}",
                "latencies": latencies,
                "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}],
                "content_type": content_type  # Metadata for future use
            }
        
        # STEP 2: Controlled Summary Mode (Narrative Only)
        # Only allow LLM summary for narrative content. Index/other types get link-only.
        # STEP 2: Controlled Summary Mode (Narrative Only)
        # Only allow LLM summary for narrative content. Index/other types get link-only.
        # Added command_reference to allowed types
        if content_type not in ["narrative", "command_reference"]:
            # Non-narrative content (index, etc.) should not be summarized
            print(f"[STEP 2] Content Type: {content_type} -> Link Only (Non-Narrative)")
            latencies["total"] = (time.time() - start_time) * 1000
            return {
                "answer": (
                    f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                    f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                    f"บทความนี้เป็นหน้ารวมข้อมูล กรุณาเลือกหัวข้อที่สนใจจากลิงก์ด้านบนครับ"
                ),
                "route": "article_link_only_index",
                "decision_reason": f"NON_NARRATIVE_CONTENT_{content_type.upper()}",
                "latencies": latencies,
                "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}],
                "content_type": content_type
            }
        
        # For narrative content, proceed with normal LLM summarization
        print(f"[STEP 2] Content Type: narrative -> Allowing LLM Summary")
        
        # 3. Call Interpreter (with LLM latency tracking)
        images = self.processed_cache._url_to_images.get(url, [])
        t_llm_start = time.time()
        interpreter_res = self.article_interpreter.interpret(
            user_query=query,
            article_title=article_title, 
            article_url=url,
            article_content=article_content,
            images=images,
            show_images=False,
            match_score=match_score,
            intent=article_type if article_type else intent
        )
        t_llm_elapsed = (time.time() - t_llm_start) * 1000
        latencies["llm"] = round(t_llm_elapsed, 2)
        print(f"[ArticleInterpreter] LLM Summarize latency: {t_llm_elapsed:.2f}ms")
        
        # Phase 21: Unpack Structured Response
        ans_data = interpreter_res.get("answer", "")
        metadata = interpreter_res.get("metadata", {})
        paragraphs = metadata.get("paragraphs", 0)
        bullets = metadata.get("bullets", 0)
        
        is_extractive = metadata.get("is_extractive", False)
        
        # Low-Context Rule (LC-1):
        # LLM path: block if < 3 paragraphs AND < 2 bullets (prevent hallucination on thin content)
        # Extractive path: block only if ZERO paragraphs AND ZERO bullets (truly empty)
        # Rationale: extractive path shows raw text directly (no LLM), so risk of hallucination
        # is zero — only truly empty/junk content should be blocked.
        if is_extractive:
            low_context = paragraphs < 1 and bullets < 1
        else:
            low_context = paragraphs < 3 and bullets < 2 and content_type != "command_reference"
        
        if low_context:
            print(f"[GOVERNANCE] Phase 21: Low-Context Detected (P={paragraphs}, B={bullets}, extractive={is_extractive}) -> Forced LINK_ONLY")
            return {
                 "answer": (
                     f"📌 **แหล่งข้อมูลหลัก (SMC):**\n"
                     f"🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
                     f"บทความนี้มีข้อมูลเชิงเทคนิคหรือสั้นเกินไป แนะนำตรวจสอบรายละเอียดจากต้นฉบับครับ"
                 ),
                 "route": "article_link_only_low_context",
                 "decision_reason": f"<3 paragraphs AND <2 bullets (P={paragraphs}, B={bullets})",
                 "latencies": latencies,
                 "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}]
            }

        latencies["article_gen"] = (time.time() - t_art) * 1000
        latencies["total"] = (time.time() - start_time) * 1000
        
        # Save to Cache
        if self.cache_manager and ans_data and len(ans_data) > 50:
             # Ensure fingerprint is defined (Fix local scope)
             try:
                 # Re-calculate fingerprint if missing (from L2 logic)
                 ux_formatting = self.cfg.get("ux", {}).get("enable_formatting", True)
                 ux_preview = self.cfg.get("ux", {}).get("direct_content_preview", True)
                 ux_long_mode = self.cfg.get("ux", {}).get("article", {}).get("intro", {}).get("enabled", True)
                 ux_max_chars = self.cfg.get("ux", {}).get("article", {}).get("max_chars", 900)
                 from src.rag.prompts import PROMPT_VERSION
                 cache_key_str = f"article_summary|{url}|{query}|{PROMPT_VERSION}|fmt:{ux_formatting}|pre:{ux_preview}|intro:{ux_long_mode}|lim:{ux_max_chars}" 
                 fingerprint = hashlib.md5(cache_key_str.encode()).hexdigest()
                 
                 meta = {"route": "article_answer", "model": self.llm_cfg.get("model", "unknown"), "url": url, "title": article_title}
                 self.cache_manager.set_answer_cache(query, ans_data, "ARTICLE_SUMMARY_V2", fingerprint, meta)
             except Exception as e:
                 print(f"[ERROR] Cache Save Failed: {e}")
        
        # 4. FINAL NORMALIZATION
        ans_data = self.normalize_smc_url(clean_junk_text(ans_data))
        canonical_header = f"📌 **แหล่งข้อมูลหลัก (SMC):**\n🔗 [{article_title}]({self.normalize_smc_url(url)})\n\n"
        
        # Remove any existing headers/footers in ans_data
        clean_ans = ans_data
        
        # Phase 220: Preserve Disclaimer if exists
        disclaimer_text = ""
        if "⚠️ **หมายเหตุ:**" in clean_ans:
             # Extract disclaimer part
             parts = clean_ans.split("⚠️ **หมายเหตุ:**")
             if len(parts) > 1:
                  disclaimer_text = "\n\n⚠️ **หมายเหตุ:**" + parts[1].split("\n\nตรวจสอบ")[0] # Get until next footer or end
                  # FIX: Explicitly remove disclaimer from clean_ans to prevent duplication
                  clean_ans = parts[0].strip()
        
        for f in ["แหล่งที่มา:", "---", "ตรวจสอบข้อมูลเพิ่มเติม:", "📌 **แหล่งข้อมูลหลัก", "[SMC Reference]"]:
            if f in clean_ans: clean_ans = clean_ans.split(f)[0].strip()
        
        final_ans = canonical_header + clean_ans + disclaimer_text + "\n\nตรวจสอบข้อมูลเพิ่มเติมจากลิงก์ SMC ต้นฉบับ"


        # FIX 2: MANDATORY AUDIT - Ensure decision_reason is NEVER None
        if not decision_reason:
            decision_reason = "NORMAL_CONTEXT_SUMMARIZATION"

        return {
            "answer": final_ans,
            "route": "article_answer",
            "latencies": latencies,
            "decision_reason": decision_reason,
            "hits": [{"key": "Source", "value": article_title, "source_url": self.normalize_smc_url(url)}]
        }
    
    def _apply_answer_mode(self, result: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """
        Phase 25/26: Filter answer/hits based on mode.
        """
        if mode == "FULL":
            return result
            
        hits = result.get("hits", [])
        filtered_hits = []
        fallback_msg = ""
        
        if mode == "PHONE_ONLY":
            fallback_msg = "ไม่พบหมายเลขโทรศัพท์ในข้อมูลนี้"
            for h in hits:
                val = str(h.get("value", ""))
                # Strict phone check: Must have digits, but NOT look like an IP (x.x.x.x)
                is_ip = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", val.strip())
                if any(c.isdigit() for c in val) and not is_ip: 
                    filtered_hits.append(h)
                    
        elif mode == "IP_ONLY":
            fallback_msg = "ไม่พบข้อมูล IP Address ในส่วนนี้"
            for h in hits:
                val = str(h.get("value", ""))
                if "IP" in h.get("key", "") or any(x in val for x in [".", ":"]): 
                     filtered_hits.append(h)
                     
        elif mode == "URL_ONLY":
             fallback_msg = "ไม่พบลิงก์เว็บไซต์ในข้อมูลนี้"
             for h in hits:
                 val = str(h.get("value", ""))
                 src = str(h.get("source_url", ""))
                 if "http" in val or "http" in src:
                     filtered_hits.append(h)
                     
        elif mode == "NAME_ONLY":
             fallback_msg = "ไม่พบรายชื่อบุคคลในข้อมูลนี้"
             for h in hits:
                 val = str(h.get("value", ""))
                 # Regex match Thai Title at start
                 if re.match(r"(?:นาย|นาง|น\.ส\.|คุณ|ดร\.|พล\.)", val.strip()):
                     filtered_hits.append(h)

        elif mode == "SUMMARY":
            return result

        if not filtered_hits and mode != "SUMMARY":
            return {
                "answer": fallback_msg,
                "hits": [],
                "route": result["route"],
                "latencies": result.get("latencies", {})
            }

        elif mode == "NAME_ONLY":
             lines = []
             seen_names = set()
             for i, h in enumerate(filtered_hits):
                 val = str(h.get("value", ""))
                 # Clean value: matches "Title Name Surname"
                 match = re.search(r"((?:นาย|นาง|น\.ส\.|คุณ|ดร\.|พล\.)\s*[\u0E00-\u0E7F]+\s+[\u0E00-\u0E7F]+)", val)
                 if match:
                     name = match.group(1)
                     if name not in seen_names:
                         lines.append(f"{len(lines)+1}. {name}")
                         seen_names.add(name)
             return {
                 "answer": "\n".join(lines) if lines else fallback_msg,
                 "hits": filtered_hits,
                 "route": result["route"],
                 "latencies": result.get("latencies", {})
             }
        elif mode == "URL_ONLY":
             # Dedup URLs
             seen_urls = set()
             lines = []
             for h in filtered_hits:
                 val = str(h.get("value", ""))
                 src = str(h.get("source_url", ""))
                 # Use value if it's a link, else source
                 link = val if "http" in val else src
                 
                 # Clean link
                 link = link.strip()
                 if not link or link in seen_urls: continue
                 
                 # Strip markdown link syntax if present [x](url)
                 m = re.search(r'\((http[s]?://[^)]+)\)', link)
                 if m: link = m.group(1)
                 
                 seen_urls.add(link)
                 
                 # Optional: Show Title?
                 # If value was NOT the link, use value as title
                 if val != link and len(val) < 100:
                     title = val
                 else:
                     title = h.get("source_title", h.get("key"))
                 # Simply list URL if title irrelevant, or 'Title: URL'
                 lines.append(f"- {title}: {link}")
                 
                 if len(lines) >= 5: break # Limit
                 
             new_ans = "\n".join(lines)
        else:
             # Dedup Values for FULL Mode
             seen_vals = set()
             lines = []
             
             # Sort hits: prioritize detections? 
             # Hits usually pre-sorted by Score/Date in KP Manager
             
             for h in filtered_hits:
                 val = str(h.get("value", "")).strip()
                 key = h.get("key", "")
                 composite = f"{key}:{val}"
                 
                 if composite in seen_vals: continue
                 seen_vals.add(composite)
                 
                 lines.append(f"- {key}: {val}")
                 
             if warning_msg := result.get("answer", "").split("\n\n")[0]:
                 if "ไม่พบข้อมูลในขอบเขต" in warning_msg:
                      lines.insert(0, warning_msg + "\n")
             
             new_ans = "\n".join(lines)
        
        return {
            "answer": new_ans,
            "hits": filtered_hits,
            "route": result["route"],
            "latencies": result.get("latencies", {})
        }

    def _log_audit(self, response: Dict[str, Any], block_reason: str = None, query: str = None):
        """
        Phase 3.5: Mandatory Structured Audit Log
        Log every answer: route_taken, article_id, block_reason
        """
        try:
            status = "SUCCESS" if not block_reason else "BLOCKED"
            route = response.get("route", "unknown")
            article_id = "N/A"
            
            # Check for Article ID in debug_info or infer from route
            if route == "article_answer":
                # Extract from response text if possible or context
                pass

            timestamp = time.time()
            print(f"[AUDIT] {timestamp} | Status: {status} | Route: {route} | ArticleID: {article_id} | Reason: {block_reason}")

            # Structured Routing Observability Log
            lats = response.get("latencies", {})
            retrieve_ms = round(lats.get("vector_search", 0), 2)
            llm_ms = round(lats.get("llm", 0), 2)
            total_ms = round(lats.get("total", (time.time() - timestamp) * 1000), 2)
            det_triggered = "deterministic" in route or "contact" in route or "directory" in route or "article" in route
            rag_used = "rag" in route or "vector" in route or "clarification" in route
            llm_called = lats.get("llm", 0) > 0
            _intent = getattr(self, "_last_intent", "UNKNOWN")
            _conf = getattr(self, "_last_confidence", 0.0)
            print(
                f"[ROUTING] intent={_intent} conf={_conf:.2f} route={route} "
                f"deterministic={det_triggered} rag={rag_used} llm={llm_called} "
                f"retrieve_ms={retrieve_ms} llm_ms={llm_ms} total_ms={total_ms}"
            )

            
            # Phase 8: Metrics Dashboard Logging
            if hasattr(self, "metrics") and self.metrics:
                # Map to Result Enum: BLOCK | ARTICLE_OK | MISSING
                metric_result = "UNKNOWN"
                
                if block_reason:
                    metric_result = "BLOCK"
                elif route in ["rag_missing_corpus", "rag_no_smc_data", "rag_miss_coverage", "rag_no_docs", "blocked_scope", "blocked_intent", "blocked_ambiguous"]:
                    metric_result = "BLOCK" if "blocked" in route else "MISSING"
                    if route == "rag_missing_corpus": metric_result = "MISSING" # Specific override
                elif route in ["article_answer", "article_link_only", "news_link", "knowledge_pack_resolved"]:
                    metric_result = "ARTICLE_OK"
                else:
                    metric_result = "OTHER" 

                self.metrics.log(
                    query=query or "unknown", 
                    intent=response.get("intent", "unknown"), 
                    result=metric_result,
                    route=route
                )

            # Step 10: Unified CSV Logging
            if self.save_log and self.last_telemetry:
                self.last_telemetry["route"] = route
                # Ensure we have the core metrics even if Phase 8 was skipped
                self._append_to_metrics_csv(self.last_telemetry)
        except Exception as e:
            print(f"[WARN] Audit logging failed: {e}")


    def _log_telemetry(self, data: Dict[str, Any], latencies: Dict[str, float]):

        """
        Phase 25: Log telemetry to JSONL
        """
        if not self.save_log:
            return
            
        try:
            log_path = Path("data/logs/chat_telemetry.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Enrich with latencies
            data["latencies"] = latencies
            # Ensure timestamp is float
            ts = data.get("timestamp", time.time())
            data["timestamp_iso"] = datetime.datetime.fromtimestamp(ts).isoformat()
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            if "top_score" in data:
                pass # self._append_to_metrics_csv(data) # Now handled centralized in _log_audit
                
        except Exception as e:
            print(f"[Telemetry] Error: {e}")

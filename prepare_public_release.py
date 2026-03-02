"""
prepare_public_release.py
=========================
Script to create a clean, sanitized public release of the RAG system.

WHAT IT DOES:
1. Copies `src/` to `public_release/src/`, filtering out:
   - `__pycache__`
   - `*.backup*`, `*.checkpoint*`
   - `tests/` and `src/testing/` (unless specific smoketests needed)
   - `data/` (creates empty structure)
2. Scans for and REPLACES sensitive strings:
   - Internal IPs: 10.192.133.33 -> 10.x.x.x
   - Internal Domains: .intra.html, .tot.co.th -> .example.com
3. Generates a generic `requirements.txt`
4. Creates a `README_PUBLIC.md` explaining how to setup.

"""

import os
import shutil
import re
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RELEASE_DIR = PROJECT_ROOT.parent / "public_release"

# Files/Dirs to EXCLUDE
EXCLUDE_PATTERNS = [
    r"__pycache__",
    r"\.DS_Store",
    r"\.git",
    r"\.env",
    r"venv",
    r"logs",
    r"backups",
    r"data/processed",
    r"data/raw",
    r"data/vectorstore",
    r"\.backup",
    r"\.checkpoint",
    r"src/testing",  # Exclude deep internal tests
    r"tests/manual",
    r"manual_update_knowledge.py", # Local script
]

# Sensitive patterns to REDACT
REPLACEMENTS = [
    (r"10\.192\.133\.33", "10.x.x.x"),
    (r"10\.235\.\d+\.\d+", "10.x.x.x"),
    (r"192\.168\.\d+\.\d+", "192.168.x.x"),
    (r"smc/index\.php", "portal/index.php"),
    (r"intra\.ntplc\.co\.th", "intra.example.com"),
    (r"tot\.co\.th", "example.com"),
    # Add explicit credential patterns if verified
]

# Whitelist of core files to definitely keep (even if they look like scripts)
KEEP_SCRIPTS = [
    "metrics_report.py",
]

def clean_and_copy(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    
    def ignore_patterns(path, names):
        ignored = set()
        for name in names:
            full_path = str(Path(path) / name)
            # Check against excludes
            for pat in EXCLUDE_PATTERNS:
                if re.search(pat, full_path):
                    ignored.add(name)
                    break
        return ignored

    shutil.copytree(src, dst, ignore=ignore_patterns)

def sanitize_file(file_path: Path):
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content
        
        for pattern, repl in REPLACEMENTS:
            content = re.sub(pattern, repl, content)
            
        if content != original:
            file_path.write_text(content, encoding="utf-8")
            print(f"  [SANITIZED] {file_path.name}")
            
    except Exception as e:
        print(f"  [WARN] Could not sanitize {file_path}: {e}")

def main():
    print(f"Preparing Public Release in: {RELEASE_DIR}")
    
    # 1. Create Structure
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir()
    
    # 2. Copy Source
    print("Copying source code...")
    clean_and_copy(PROJECT_ROOT / "src", RELEASE_DIR / "src")
    
    # Copy main entry points
    for f in ["requirements.txt", "README.md", "pyproject.toml"]:
        if (PROJECT_ROOT / f).exists():
            shutil.copy2(PROJECT_ROOT / f, RELEASE_DIR / f)
            
    # Create empty data dirs
    (RELEASE_DIR / "data" / "raw").mkdir(parents=True)
    (RELEASE_DIR / "data" / "processed").mkdir(parents=True)
    (RELEASE_DIR / "data" / "vectorstore").mkdir(parents=True)
    
    # 3. Sanitize
    print("Sanitizing code...")
    for root, dirs, files in os.walk(RELEASE_DIR):
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
                sanitize_file(Path(root) / file)
                
    # 4. Create Public README
    readme_content = """# RAG System (Public Release)

This is a sanitized version of the RAG system, designed for local deployment and testing.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize data directories:
   Ensure `data/raw`, `data/processed`, and `data/vectorstore` exist.

3. Run the API:
   ```bash
   python -m src.main api
   ```

## Note on Data
Real organizational data has been removed. You will need to ingest your own documents using `src/ingest/` scripts.
"""
    (RELEASE_DIR / "README_PUBLIC.md").write_text(readme_content)
    
    print("\n✅ Public Release Prepared Successfully!")
    print(f"Location: {RELEASE_DIR}")

if __name__ == "__main__":
    main()

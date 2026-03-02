#!/bin/bash
# Direct sed replacement for vendor list generation

FILE="/Users/jakkapatmac/Documents/NT/RAG/rag_web/src/chat_engine.py"

# Create backup
cp "$FILE" "${FILE}.backup_numeric"

# Replace lines 2214-2227 (category grouping) with numeric session
sed -i.tmp '2214,2227c\
                         # ============================================================\
                          # NUMERIC SELECTION: Global numbering (NO category reset)\
                          # ============================================================\
                          items = []\
                          for art in vendor_articles:\
                              items.append({\
                                  "url": art['"'"'url'"'"'],\
                                  "title": art['"'"'title'"'"'],\
                                  "category": art.get('"'"'category'"'"', '"'"''"'"')\
                              })\
                          \
                          # Create session with global numbering\
                          session = self.numeric_selection_resolver.create_session(\
                              items, context="article_selection",\
                              prompt_text=f"กรุณาเลือกหมายเลข (1-{len(items)})"\
                          )\
                          options_text = self.numeric_selection_resolver.format_numbered_list(\
                              session['"'"'items'"'"'], context="article_selection"\
                          )\
                          self.pending_numeric_session = session
' "$FILE"

# Update answer format (lines 2231-2233)
sed -i.tmp2 '2231,2233c\
                                  f"พบ {len(vendor_articles)} เอกสารที่เกี่ยวข้อง:\\n\\n"\
                                  f"{options_text}\\n\\n"\
                                  f"{session['"'"'prompt_text'"'"']}"
' "$FILE"

# Clean up temp files
rm -f "${FILE}.tmp" "${FILE}.tmp2"

echo "✅ Vendor list updated with numeric selection"
echo "📝 Backup saved: ${FILE}.backup_numeric"
echo ""
echo "🧪 Test now:"
echo "   python3 -m src.main chat"
echo "   Query: คำสั่ง huawei"
echo "   Expected: 1. config VAS... 2. How to checking..."

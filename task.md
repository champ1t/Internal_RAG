# Enhancing RAG Robustness

- [x] **Robust Thai Normalization**
    - [x] Handle Thai digits
    - [x] Handle zero-width characters
    - [x] Handle common typos and spacing
- [ ] **Guided Contact Fallback**
    - [ ] Debug `ContactHandler` kill-switch logic
    - [ ] Ensure `DirectoryHandler` fallback is triggered for location queries
    - [ ] Verify helpful suggestions instead of "MISS"
- [ ] **Web Knowledge Route**
    - [ ] Implement `WebHandler` with search capability
    - [x] Integrate `strip_navigation_text` into `src/rag/article_interpreter.py` (Phase 173 block) <!-- id: 5 -->
- [x] Fix `smart_truncate` in `src/rag/article_cleaner.py` to append footer <!-- id: 6 --> [ ] Handle missing API keys gracefully
- [ ] **Junk Link Filtering**
    - [x] Add extension checking in `RetrievalOptimizer`
    - [x] Apply `deduplicate_paragraphs` in `article_interpreter.py` <!-- id: 7 -->verification step, need final confirmation)
- [x] **Regression Testing**
    - [ ] Run full 8-item regression suite
    - [ ] Verify "Bridge Port", "North Flood News", "Hat Yai Contact" cases

- [/] **Test "Web Hijack Prevention"**: Query "config vlan cisco" (ensure routed to `GENERAL_QA`) <!-- id: 9 -->
- [ ] **Test "Legitimate Web Query"**: Query "news about Chiang Rai floods" (ensure routed to `WEB_KNOWLEDGE`) <!-- id: 10 -->
- [ ] **Test "Article Presentation"**: Verify "Read More" link and clean text <!-- id: 11 --> [ ] **Regression Defense Strategy**
        - [x] 1. Freeze "Decision Policy" (`config/routing_policy_v1.yaml`)
        - [ ] 2. Lock Regression Tests (`tests/regression_guard.py`)
        - [ ] 3. CLARIFY = Fallback Only (Enforce ordering)
        - [ ] 4. Safety Check Logging (Structured logs)
        - [ ] 5. Feature Declaration Checklist (`governance/REGRESSION_SURFACE_CHECKLIST.md`)
        - [ ] 6. Tag + Branch (`v1.0-phase175`)

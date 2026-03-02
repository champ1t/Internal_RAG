#!/bin/bash
cd /Users/jakkapatmac/Documents/NT/RAG/rag_web

echo "=========================================="
echo "PHASE 21 & 22 FINAL VERIFICATION"
echo "=========================================="

run_test() {
    local id="$1"
    local query="$2"
    echo ""
    echo "[$id] Query: $query"
    output=$(echo "$query" | timeout 120 python3 -m src.main chat 2>&1)
    
    # Extract route
    route=$(echo "$output" | grep -oE "route=[a-z_]+" | head -1 | cut -d= -f2)
    if [ -z "$route" ]; then
        route=$(echo "$output" | grep "confidence_mode" | head -1 | sed 's/.*confidence_mode": "\(.*\)".*/\1/')
    fi
    
    # Extract decision reason
    reason=$(echo "$output" | grep "decision_reason" | head -1 | sed 's/.*decision_reason": "\(.*\)".*/\1/')
    
    echo "  Route: $route"
    echo "  Reason: $reason"
}

# 1) Exact Match
run_test "TC-V21-1" "zte sw command"

# 2) Soft Exact Match
run_test "TC-V21-2" "ZTE--SW--Command"

# 3) Normal Article (High context)
run_test "TC-V21-3" "GPON Overview"

# 4) Non-SMC (Cisco OLT)
run_test "TC-V22-1" "Cisco OLT new command 2024"

# 5) Out-of-Scope Vendor (Juniper)
run_test "TC-V22-2" "Juniper router config"

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="

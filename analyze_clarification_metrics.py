import pandas as pd
import os
import sys

def analyze_metrics(csv_path="results/clarification_metrics_log.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV is empty.")
        return

    total_queries = len(df)
    clarify_df = df[df['clarification_triggered'] == True]
    clarification_count = len(clarify_df)
    
    clarification_rate = (clarification_count / total_queries) * 100 if total_queries > 0 else 0
    avg_score_gap_all = df['score_gap'].mean()
    avg_score_gap_clarified = clarify_df['score_gap'].mean()
    
    # Target Validation
    # Target Clarification Rate: 10-25%
    # Target Avg Score Gap: < 0.07 when triggered
    
    rate_ok = 10 <= clarification_rate <= 25
    gap_ok = avg_score_gap_clarified < 0.07 if not pd.isna(avg_score_gap_clarified) else True

    print("="*60)
    print("CLARIFICATION STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Total Queries Analysis:     {total_queries}")
    print(f"Clarification Triggered:    {clarification_count}")
    print(f"Clarification Rate:         {clarification_rate:.2f}% (Target: 10-25%)")
    print(f"Average Score Gap (All):    {avg_score_gap_all:.4f}")
    print(f"Average Score Gap (Clarify): {avg_score_gap_clarified:.4f} (Target: < 0.07)")
    
    print("-"*60)
    print("HEALTH CHECK")
    print(f"Rate within bounds:         {'✅ PASS' if rate_ok else '⚠️ OUT OF BOUNDS'}")
    print(f"Score gap healthy:           {'✅ PASS' if gap_ok else '⚠️ HIGH AMBIGUITY'}")
    
    # Route Distribution
    print("-"*60)
    print("ROUTE DISTRIBUTION")
    print(df['route'].value_counts())
    
    # Reason Distribution
    print("-"*60)
    print("CLARIFICATION REASONS")
    print(clarify_df['clarification_reason'].value_counts())
    
    print("="*60)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/clarification_metrics_log.csv"
    analyze_metrics(path)

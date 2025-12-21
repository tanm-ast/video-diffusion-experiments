"""
Analyze Ablation Results: Per-Video Tabular Comparison

Loads results.json and generates:
1. Per-video comparison tables (printed)
2. CSV files for each video
3. Summary CSV with all data

Usage:
    python analyze_ablation_results.py
"""

import json
import pandas as pd
from pathlib import Path

# ============================================================
# Load Data
# ============================================================

def load_results(json_path: str = "results.json") -> list:
    with open(json_path, 'r') as f:
        return json.load(f)


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert results JSON to a flat DataFrame."""
    rows = []
    for r in results:
        row = {
            "video_name": r["config"]["video_name"],
            "hypothesis": r["config"]["hypothesis"],
            "experiment_id": r["config"]["experiment_id"],
            "guidance_scale": r["config"]["guidance_scale"],
            "num_inference_steps": r["config"]["num_inference_steps"],
            "mse_mean": r["metrics"]["mse_mean"],
            "mse_std": r["metrics"]["mse_std"],
            "lpips_mean": r["metrics"]["lpips_mean"],
            "lpips_std": r["metrics"]["lpips_std"],
            "flow_mean": r["metrics"]["flow_mean"],
            "flow_variance": r["metrics"]["flow_variance"],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


# ============================================================
# Analysis Functions
# ============================================================

def compute_improvements(df: pd.DataFrame, metric: str, lower_is_better: bool = True) -> pd.DataFrame:
    """Add improvement percentage column relative to baseline."""
    result = df.copy()
    
    for video in df["video_name"].unique():
        video_mask = df["video_name"] == video
        baseline_mask = video_mask & (df["hypothesis"] == "baseline")
        
        if baseline_mask.sum() == 0:
            continue
            
        baseline_val = df.loc[baseline_mask, metric].values[0]
        
        if lower_is_better:
            improvement = ((baseline_val - df.loc[video_mask, metric]) / baseline_val) * 100
        else:
            improvement = ((df.loc[video_mask, metric] - baseline_val) / baseline_val) * 100
        
        result.loc[video_mask, f"{metric}_improvement"] = improvement
    
    return result


def get_video_table(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Get formatted table for a single video."""
    video_df = df[df["video_name"] == video_name].copy()
    
    # Define column order and formatting
    display_cols = [
        "hypothesis",
        "guidance_scale",
        "num_inference_steps",
        "mse_mean",
        "mse_improvement",
        "lpips_mean", 
        "lpips_improvement",
        "flow_variance",
        "flow_improvement",
    ]
    
    # Compute improvements
    video_df = compute_improvements(video_df, "mse_mean", lower_is_better=True)
    video_df = compute_improvements(video_df, "lpips_mean", lower_is_better=True)
    video_df = compute_improvements(video_df, "flow_variance", lower_is_better=True)
    
    # Rename for display
    video_df = video_df.rename(columns={
        "flow_variance_improvement": "flow_improvement"
    })
    
    # Sort by hypothesis for consistent ordering
    hypothesis_order = ["baseline", "cfg_only", "steps_only", "prompt_only", "recommended"]
    video_df["sort_order"] = video_df["hypothesis"].map(
        {h: i for i, h in enumerate(hypothesis_order)}
    )
    video_df = video_df.sort_values("sort_order").drop(columns=["sort_order"])
    
    # Select and order columns
    available_cols = [c for c in display_cols if c in video_df.columns]
    return video_df[available_cols].reset_index(drop=True)


def print_video_table(df: pd.DataFrame, video_name: str):
    """Print formatted table for a video."""
    table = get_video_table(df, video_name)
    
    print(f"\n{'='*90}")
    print(f" {video_name.upper()}")
    print(f"{'='*90}")
    
    # Format for printing
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 100 else f'{x:.1f}')
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 15)
    
    # Rename columns for display
    display_table = table.rename(columns={
        "hypothesis": "Intervention",
        "guidance_scale": "CFG",
        "num_inference_steps": "Steps",
        "mse_mean": "MSE",
        "mse_improvement": "MSE Δ%",
        "lpips_mean": "LPIPS",
        "lpips_improvement": "LPIPS Δ%",
        "flow_variance": "Flow Var",
        "flow_improvement": "Flow Δ%",
    })
    
    print(display_table.to_string(index=False))
    
    # Find best intervention
    non_baseline = table[table["hypothesis"] != "baseline"]
    if len(non_baseline) > 0:
        best_mse = non_baseline.loc[non_baseline["mse_mean"].idxmin(), "hypothesis"]
        best_lpips = non_baseline.loc[non_baseline["lpips_mean"].idxmin(), "hypothesis"]
        best_flow = non_baseline.loc[non_baseline["flow_variance"].idxmin(), "hypothesis"]
        
        print(f"\n  Best by MSE:      {best_mse}")
        print(f"  Best by LPIPS:    {best_lpips}")
        print(f"  Best by Flow Var: {best_flow}")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary showing best intervention per video."""
    summary_rows = []
    
    for video in df["video_name"].unique():
        video_df = df[df["video_name"] == video].copy()
        baseline = video_df[video_df["hypothesis"] == "baseline"]
        non_baseline = video_df[video_df["hypothesis"] != "baseline"]
        
        if len(baseline) == 0 or len(non_baseline) == 0:
            continue
        
        baseline_mse = baseline["mse_mean"].values[0]
        baseline_lpips = baseline["lpips_mean"].values[0]
        baseline_flow = baseline["flow_variance"].values[0]
        
        # Find best for each metric
        best_mse_row = non_baseline.loc[non_baseline["mse_mean"].idxmin()]
        best_lpips_row = non_baseline.loc[non_baseline["lpips_mean"].idxmin()]
        best_flow_row = non_baseline.loc[non_baseline["flow_variance"].idxmin()]
        
        mse_improvement = (baseline_mse - best_mse_row["mse_mean"]) / baseline_mse * 100
        lpips_improvement = (baseline_lpips - best_lpips_row["lpips_mean"]) / baseline_lpips * 100
        flow_improvement = (baseline_flow - best_flow_row["flow_variance"]) / baseline_flow * 100
        
        summary_rows.append({
            "Video": video,
            "Best MSE": best_mse_row["hypothesis"],
            "MSE  delta %": f"+{mse_improvement:.1f}%",
            "Best LPIPS": best_lpips_row["hypothesis"],
            "LPIPS delta %": f"+{lpips_improvement:.1f}%",
            "Best Flow": best_flow_row["hypothesis"],
            "Flow delta %": f"+{flow_improvement:.1f}%",
        })
    
    return pd.DataFrame(summary_rows)


def compute_intervention_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average improvement per intervention across all videos."""
    df_with_improvements = df.copy()
    
    # Add improvements for all metrics
    df_with_improvements = compute_improvements(df_with_improvements, "mse_mean")
    df_with_improvements = compute_improvements(df_with_improvements, "lpips_mean")
    df_with_improvements = compute_improvements(df_with_improvements, "flow_variance")
    
    # Group by hypothesis and average
    avg_df = df_with_improvements.groupby("hypothesis").agg({
        "mse_mean_improvement": "mean",
        "lpips_mean_improvement": "mean",
        "flow_variance_improvement": "mean",
    }).reset_index()
    
    avg_df.columns = ["Intervention", "Avg MSE delta %", "Avg LPIPS delta %", "Avg Flow delta %"]
    
    # Sort by intervention order
    hypothesis_order = ["baseline", "cfg_only", "steps_only", "prompt_only", "recommended"]
    avg_df["sort"] = avg_df["Intervention"].map({h: i for i, h in enumerate(hypothesis_order)})
    avg_df = avg_df.sort_values("sort").drop(columns=["sort"])
    
    return avg_df


# ============================================================
# CSV Export
# ============================================================

def save_all_csvs(df: pd.DataFrame, output_dir: str = "outputs/07_analysis"):
    """Save per-video CSVs and summary CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Per-video CSVs
    for video in df["video_name"].unique():
        table = get_video_table(df, video)
        csv_path = output_path / f"{video}_ablation_results.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    # Full data CSV
    full_path = output_path / "all_ablation_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")
    
    # Summary CSV
    summary = generate_summary_table(df)
    summary_path = output_path / "best_interventions_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # Intervention averages CSV
    averages = compute_intervention_averages(df)
    avg_path = output_path / "intervention_averages.csv"
    averages.to_csv(avg_path, index=False)
    print(f"Saved: {avg_path}")


# ============================================================
# Main
# ============================================================

def main():
    # Load data
    results = load_results("outputs/05_ablation/results.json")
    df = results_to_dataframe(results)
    
    print("\n" + "="*90)
    print(" ABLATION STUDY RESULTS ANALYSIS")
    print("="*90)
    print(f"\nLoaded {len(results)} experiments across {df['video_name'].nunique()} videos")
    
    # Print per-video tables
    for video in sorted(df["video_name"].unique()):
        print_video_table(df, video)
    
    # Print summary
    print(f"\n\n{'='*90}")
    print(" SUMMARY: BEST INTERVENTION PER VIDEO")
    print("="*90 + "\n")
    
    summary = generate_summary_table(df)
    print(summary.to_string(index=False))
    
    # Print intervention averages
    print(f"\n\n{'='*90}")
    print(" AVERAGE IMPROVEMENT BY INTERVENTION")
    print("="*90 + "\n")
    
    averages = compute_intervention_averages(df)
    print(averages.to_string(index=False))
    
    # Winner
    non_baseline_avg = averages[averages["Intervention"] != "baseline"]
    if len(non_baseline_avg) > 0:
        best_intervention = non_baseline_avg.loc[non_baseline_avg["Avg MSE delta %"].idxmax(), "Intervention"]
        best_mse_avg = non_baseline_avg["Avg MSE delta %"].max()
        print(f"\n\n WINNER: {best_intervention} (avg MSE improvement: {best_mse_avg:.1f}%)")
    
    # Save CSVs
    print(f"\n\n{'='*90}")
    print(" SAVING CSV FILES")
    print("="*90 + "\n")
    
    save_all_csvs(df)
    
    print("\n Analysis complete!")


if __name__ == "__main__":
    main()

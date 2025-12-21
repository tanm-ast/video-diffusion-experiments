"""
Experiment 07: Analyze Grid Search Results

Analyzes the systematic grid search to find optimal hyperparameters.

Usage:
    python experiments/07_analyze_grid_search.py
    python experiments/07_analyze_grid_search.py --input path/to/results.json

Outputs:
    - Per-video CFG sweep tables
    - Per-video Steps sweep tables  
    - Prompt comparison tables
    - Optimal values summary
    - CSV files for all tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("outputs/06_grid_search_metrics/grid_search_results.json")
OUTPUT_DIR = Path("outputs/07_grid_search_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "mean_mse", 
    "std_mse", 
    "mean_lpips", 
    "std_lpips",
    "mean_flow_magnitude",
    "flow_magnitude_variance", 
    "mean_warp_error",
    "warp_error_variance",
    "flicker_index"
]

# Primary metrics for optimization (lower is better for all)
PRIMARY_METRICS = [
    "mean_mse", 
    "mean_lpips", 
    "mean_flow_magnitude",      # Amount of motion (context, lower = more static)
    "flow_magnitude_variance",  # Motion consistency (lower = smoother)
    "mean_warp_error",          # Motion predictability (lower = more predictable)
    "warp_error_variance",      # Consistency of predictability (lower = more consistent)
    "flicker_index"             # Oscillation detection (lower = less flicker)
]


# ============================================================
# Data Loading
# ============================================================

def load_results(json_path: Path) -> pd.DataFrame:
    """Load results JSON and convert to DataFrame."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    rows = []
    for r in results:
        row = {
            "video_name": r["video_name"],
            "experiment_id": r["experiment_id"],
            "cfg": r["guidance_scale"],
            "steps": r["num_inference_steps"],
            "phase": r["phase"],
            # MSE
            "mean_mse": r["mean_mse"],
            "std_mse": r["std_mse"],
            # LPIPS
            "mean_lpips": r["mean_lpips"],
            "std_lpips": r["std_lpips"],
            # Flow
            "mean_flow_magnitude": r["mean_flow_magnitude"],
            "flow_magnitude_variance": r["flow_magnitude_variance"],
            # Warp error
            "mean_warp_error": r.get("mean_warp_error"),
            "warp_error_variance": r.get("warp_error_variance"),
            # Composite
            "temporal_consistency_score": r["temporal_consistency_score"],
            "flicker_index": r["flicker_index"],
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================
# Analysis Functions
# ============================================================

def get_cfg_sweep(df: pd.DataFrame, video_name: str, fixed_steps: int = 25) -> pd.DataFrame:
    """Get CFG sweep for a video (varying CFG, fixed steps, excluding prompt ablation)."""
    mask = (
        (df["video_name"] == video_name) & 
        (df["steps"] == fixed_steps) &
        (~df["phase"].isin(["prompt_ablation"]))  # Exclude prompt experiments
    )
    sweep = df[mask].drop_duplicates(subset=["cfg"]).sort_values("cfg").copy()
    return sweep


def get_steps_sweep(df: pd.DataFrame, video_name: str, fixed_cfg: float = 7.5) -> pd.DataFrame:
    """Get Steps sweep for a video (varying steps, fixed CFG, excluding prompt ablation)."""
    mask = (
        (df["video_name"] == video_name) & 
        (df["cfg"] == fixed_cfg) &
        (~df["phase"].isin(["prompt_ablation"]))  # Exclude prompt experiments
    )
    sweep = df[mask].drop_duplicates(subset=["steps"]).sort_values("steps").copy()
    return sweep


def get_prompt_comparison(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Get prompt comparison for a video."""
    mask = (df["video_name"] == video_name) & (df["phase"] == "prompt_ablation")
    comparison = df[mask].copy()
    
    if comparison.empty:
        baseline_mask = (df["video_name"] == video_name) & (df["experiment_id"].str.contains("prompt_baseline"))
        enhanced_mask = (df["video_name"] == video_name) & (df["experiment_id"].str.contains("prompt_enhanced"))
        comparison = pd.concat([df[baseline_mask], df[enhanced_mask]])
    
    return comparison


def find_optimal(sweep: pd.DataFrame, metric: str, lower_is_better: bool = True) -> Dict:
    """Find optimal value for a metric in a sweep."""
    if sweep.empty or metric not in sweep.columns:
        return {"value": None, "param_value": None}
    
    valid = sweep.dropna(subset=[metric])
    if valid.empty:
        return {"value": None, "param_value": None}
    
    if lower_is_better:
        best_idx = valid[metric].idxmin()
    else:
        best_idx = valid[metric].idxmax()
    
    best_row = valid.loc[best_idx]
    
    if valid["cfg"].nunique() > 1:
        param_name = "cfg"
    else:
        param_name = "steps"
    
    return {
        "value": best_row[metric],
        "param_value": best_row[param_name],
        "param_name": param_name,
    }


def compute_relative_change(sweep: pd.DataFrame, metric: str, baseline_value: float) -> pd.Series:
    """Compute % change relative to baseline (positive = improvement for lower-is-better)."""
    if baseline_value == 0:
        return pd.Series([0.0] * len(sweep), index=sweep.index)
    return ((baseline_value - sweep[metric]) / baseline_value * 100)


# ============================================================
# Table Generation
# ============================================================

def generate_cfg_table(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Generate CFG sweep table for a video."""
    sweep = get_cfg_sweep(df, video_name)
    
    if sweep.empty:
        return pd.DataFrame()
    
    baseline = sweep[sweep["cfg"] == 7.5]
    
    display_cols = [
        "cfg", 
        "mean_mse", 
        "mean_lpips", 
        "mean_flow_magnitude",
        "flow_magnitude_variance", 
        "mean_warp_error", 
        "warp_error_variance",
        "flicker_index"
    ]
    available_cols = [c for c in display_cols if c in sweep.columns]
    table = sweep[available_cols].copy()
    
    if not baseline.empty:
        for metric in PRIMARY_METRICS:
            if metric in sweep.columns and baseline[metric].notna().any():
                baseline_val = baseline[metric].values[0]
                table[f"{metric}_delta"] = compute_relative_change(sweep, metric, baseline_val)
    
    return table


def generate_steps_table(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Generate Steps sweep table for a video."""
    sweep = get_steps_sweep(df, video_name)
    
    if sweep.empty:
        return pd.DataFrame()
    
    baseline = sweep[sweep["steps"] == 25]
    
    display_cols = [
        "steps", 
        "mean_mse", 
        "mean_lpips", 
        "mean_flow_magnitude",
        "flow_magnitude_variance", 
        "mean_warp_error", 
        "warp_error_variance",
        "flicker_index"
    ]
    available_cols = [c for c in display_cols if c in sweep.columns]
    table = sweep[available_cols].copy()
    
    if not baseline.empty:
        for metric in PRIMARY_METRICS:
            if metric in sweep.columns and baseline[metric].notna().any():
                baseline_val = baseline[metric].values[0]
                table[f"{metric}_delta"] = compute_relative_change(sweep, metric, baseline_val)
    
    return table


def generate_prompt_table(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Generate prompt comparison table for a video."""
    comparison = get_prompt_comparison(df, video_name)
    
    if comparison.empty:
        return pd.DataFrame()
    
    comparison = comparison.copy()
    comparison["prompt_type"] = comparison["experiment_id"].apply(
        lambda x: "enhanced" if "enhanced" in x else "baseline"
    )
    
    display_cols = [
        "prompt_type", 
        "mean_mse", 
        "mean_lpips", 
        "mean_flow_magnitude",
        "flow_magnitude_variance", 
        "mean_warp_error", 
        "warp_error_variance",
        "flicker_index"
    ]
    available_cols = [c for c in display_cols if c in comparison.columns]
    table = comparison[available_cols].copy()
    
    baseline = comparison[comparison["prompt_type"] == "baseline"]
    
    if not baseline.empty:
        for metric in PRIMARY_METRICS:
            if metric in comparison.columns and baseline[metric].notna().any():
                baseline_val = baseline[metric].values[0]
                table[f"{metric}_delta"] = compute_relative_change(comparison, metric, baseline_val)
    
    return table


def generate_optimal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary of optimal values per video."""
    videos = df["video_name"].unique()
    
    rows = []
    for video in sorted(videos):
        row = {"video": video}
        
        cfg_sweep = get_cfg_sweep(df, video)
        for metric in PRIMARY_METRICS:
            opt = find_optimal(cfg_sweep, metric)
            row[f"best_cfg_{metric}"] = opt["param_value"]
        
        steps_sweep = get_steps_sweep(df, video)
        for metric in PRIMARY_METRICS:
            opt = find_optimal(steps_sweep, metric)
            row[f"best_steps_{metric}"] = opt["param_value"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_prompt_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary of prompt improvement per video."""
    videos = df["video_name"].unique()
    rows = []
    
    for video in sorted(videos):
        comparison = get_prompt_comparison(df, video)
        if comparison.empty:
            continue
        
        baseline = comparison[comparison["experiment_id"].str.contains("baseline")]
        enhanced = comparison[comparison["experiment_id"].str.contains("enhanced")]
        
        if baseline.empty or enhanced.empty:
            continue
        
        row = {"video": video}
        
        for metric in PRIMARY_METRICS:
            if metric in baseline.columns:
                b_val = baseline[metric].values[0]
                e_val = enhanced[metric].values[0]
                
                if b_val is not None and e_val is not None and b_val != 0:
                    improvement = (b_val - e_val) / b_val * 100
                    row[f"{metric}_baseline"] = b_val
                    row[f"{metric}_enhanced"] = e_val
                    row[f"{metric}_improvement"] = improvement
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_aggregated_analysis(df: pd.DataFrame) -> Dict:
    """Aggregate analysis across all videos."""
    videos = df["video_name"].unique()
    
    cfg_wins = {metric: {} for metric in PRIMARY_METRICS}
    
    for video in videos:
        cfg_sweep = get_cfg_sweep(df, video)
        for metric in cfg_wins.keys():
            opt = find_optimal(cfg_sweep, metric)
            if opt["param_value"] is not None:
                cfg_val = opt["param_value"]
                cfg_wins[metric][cfg_val] = cfg_wins[metric].get(cfg_val, 0) + 1
    
    steps_wins = {metric: {} for metric in PRIMARY_METRICS}
    
    for video in videos:
        steps_sweep = get_steps_sweep(df, video)
        for metric in steps_wins.keys():
            opt = find_optimal(steps_sweep, metric)
            if opt["param_value"] is not None:
                steps_val = opt["param_value"]
                steps_wins[metric][steps_val] = steps_wins[metric].get(steps_val, 0) + 1
    
    return {
        "cfg_wins_by_metric": cfg_wins,
        "steps_wins_by_metric": steps_wins,
    }


# ============================================================
# Printing & Saving
# ============================================================

def print_sweep_table(table: pd.DataFrame, title: str, param_col: str):
    """Print a formatted sweep table."""
    print(f"\n{title}")
    print("-" * 80)
    
    if table.empty:
        print("  No data available")
        return
    
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 10 else f'{x:.1f}')
    pd.set_option('display.width', 200)
    
    display_cols = [c for c in table.columns if "_delta" not in c]
    print(table[display_cols].to_string(index=False))
    
    delta_cols = [c for c in table.columns if "_delta" in c]
    if delta_cols:
        default_val = '7.5' if param_col == 'cfg' else '25'
        print(f"\n  % Change from baseline ({param_col}={default_val}):")
        for _, row in table.iterrows():
            param_val = row[param_col]
            deltas = [f"{c.replace('_delta', '').replace('mean_', '')}: {row[c]:+.1f}%" 
                      for c in delta_cols if pd.notna(row[c]) and row[c] != 0]
            if deltas:
                print(f"    {param_col}={param_val}: {', '.join(deltas)}")


def save_all_csvs(df: pd.DataFrame, output_dir: Path):
    """Save all analysis tables to CSV."""
    videos = df["video_name"].unique()
    
    for video in sorted(videos):
        cfg_table = generate_cfg_table(df, video)
        if not cfg_table.empty:
            cfg_path = output_dir / f"{video}_cfg_sweep.csv"
            cfg_table.to_csv(cfg_path, index=False)
            print(f"Saved: {cfg_path}")
        
        steps_table = generate_steps_table(df, video)
        if not steps_table.empty:
            steps_path = output_dir / f"{video}_steps_sweep.csv"
            steps_table.to_csv(steps_path, index=False)
            print(f"Saved: {steps_path}")
        
        prompt_table = generate_prompt_table(df, video)
        if not prompt_table.empty:
            prompt_path = output_dir / f"{video}_prompt_comparison.csv"
            prompt_table.to_csv(prompt_path, index=False)
            print(f"Saved: {prompt_path}")
    
    optimal = generate_optimal_summary(df)
    optimal_path = output_dir / "optimal_values_summary.csv"
    optimal.to_csv(optimal_path, index=False)
    print(f"Saved: {optimal_path}")
    
    prompt_summary = generate_prompt_summary(df)
    if not prompt_summary.empty:
        prompt_summary_path = output_dir / "prompt_improvement_summary.csv"
        prompt_summary.to_csv(prompt_summary_path, index=False)
        print(f"Saved: {prompt_summary_path}")
    
    full_path = output_dir / "all_grid_search_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Grid Search Results")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Run experiments/06_measure_grid_search.py first")
        return
    
    print("Loading results...")
    df = load_results(input_path)
    print(f"Loaded {len(df)} experiments across {df['video_name'].nunique()} videos")
    
    videos = sorted(df["video_name"].unique())
    
    for video in videos:
        print(f"\n{'='*80}")
        print(f" {video.upper()}")
        print(f"{'='*80}")
        
        cfg_table = generate_cfg_table(df, video)
        print_sweep_table(cfg_table, "CFG Sweep (Steps=25)", "cfg")
        
        steps_table = generate_steps_table(df, video)
        print_sweep_table(steps_table, "Steps Sweep (CFG=7.5)", "steps")
        
        prompt_table = generate_prompt_table(df, video)
        if not prompt_table.empty:
            print(f"\n  Prompt Comparison (CFG=7.5, Steps=25)")
            print("  " + "-" * 70)
            print(prompt_table.to_string(index=False))
        
        print("\n  OPTIMAL VALUES:")
        cfg_sweep = get_cfg_sweep(df, video)
        steps_sweep = get_steps_sweep(df, video)
        
        for metric in PRIMARY_METRICS:
            cfg_opt = find_optimal(cfg_sweep, metric)
            steps_opt = find_optimal(steps_sweep, metric)
            metric_short = metric.replace("mean_", "").replace("_", " ")
            print(f"    {metric_short:<20}: Best CFG={cfg_opt['param_value']}, Best Steps={steps_opt['param_value']}")
    
    print(f"\n\n{'='*80}")
    print(" AGGREGATED ANALYSIS: OPTIMAL VALUES ACROSS ALL VIDEOS")
    print(f"{'='*80}")
    
    agg = generate_aggregated_analysis(df)
    
    print("\nCFG wins by metric (how many videos had this CFG as optimal):")
    for metric, wins in agg["cfg_wins_by_metric"].items():
        metric_short = metric.replace("mean_", "").replace("_", " ")
        wins_str = ", ".join([f"CFG {k}: {v}" for k, v in sorted(wins.items())])
        print(f"  {metric_short:<20}: {wins_str}")
    
    print("\nSteps wins by metric:")
    for metric, wins in agg["steps_wins_by_metric"].items():
        metric_short = metric.replace("mean_", "").replace("_", " ")
        wins_str = ", ".join([f"Steps {int(k)}: {v}" for k, v in sorted(wins.items())])
        print(f"  {metric_short:<20}: {wins_str}")
    
    print("\n" + "=" * 80)
    print(" RECOMMENDATIONS")
    print("=" * 80)
    
    cfg_total_wins = {}
    for metric_wins in agg["cfg_wins_by_metric"].values():
        for cfg, count in metric_wins.items():
            cfg_total_wins[cfg] = cfg_total_wins.get(cfg, 0) + count
    
    if cfg_total_wins:
        best_cfg = max(cfg_total_wins.items(), key=lambda x: x[1])
        print(f"\nMost commonly optimal CFG: {best_cfg[0]} ({best_cfg[1]} wins across all metrics)")
    
    steps_total_wins = {}
    for metric_wins in agg["steps_wins_by_metric"].values():
        for steps, count in metric_wins.items():
            steps_total_wins[steps] = steps_total_wins.get(steps, 0) + count
    
    if steps_total_wins:
        best_steps = max(steps_total_wins.items(), key=lambda x: x[1])
        print(f"Most commonly optimal Steps: {int(best_steps[0])} ({best_steps[1]} wins across all metrics)")
    
    # Prompt improvement summary
    prompt_summary = generate_prompt_summary(df)
    if not prompt_summary.empty:
        print("\n" + "=" * 80)
        print(" PROMPT ENGINEERING IMPACT")
        print("=" * 80)
        print("\n  (Positive % = enhanced is better, Negative % = enhanced is worse)")
        
        # Per-video summary first
        print("\n  Per-Video Results:")
        print("  " + "-" * 76)
        print(f"  {'Video':<20} {'MSE':<12} {'LPIPS':<12} {'Flicker':<12} {'Verdict':<15}")
        print("  " + "-" * 76)
        
        for _, row in prompt_summary.iterrows():
            video = row["video"]
            
            # Get key metrics
            mse_imp = row.get("mean_mse_improvement", 0)
            lpips_imp = row.get("mean_lpips_improvement", 0)
            flicker_imp = row.get("flicker_index_improvement", 0)
            
            # Count wins across all metrics
            wins = sum(1 for m in PRIMARY_METRICS 
                      if f"{m}_improvement" in row and row[f"{m}_improvement"] > 5)
            losses = sum(1 for m in PRIMARY_METRICS 
                        if f"{m}_improvement" in row and row[f"{m}_improvement"] < -5)
            
            if wins > losses:
                verdict = "✓ HELPS"
            elif losses > wins:
                verdict = "✗ HURTS"
            else:
                verdict = "~ Neutral"
            
            print(f"  {video:<20} {mse_imp:+8.1f}%    {lpips_imp:+8.1f}%    {flicker_imp:+8.1f}%    {verdict}")
        
        print("  " + "-" * 76)
        
        # Metric-level summary
        print("\n  Metric-Level Summary:")
        print("  " + "-" * 76)
        
        for metric in PRIMARY_METRICS:
            imp_col = f"{metric}_improvement"
            if imp_col in prompt_summary.columns:
                values = prompt_summary[imp_col].dropna()
                
                # Count wins/losses
                wins = (values > 5).sum()      # >5% improvement
                losses = (values < -5).sum()   # >5% worse
                neutral = len(values) - wins - losses
                
                # Use median (robust to outliers) instead of mean
                median_imp = values.median()
                
                metric_short = metric.replace("mean_", "").replace("_", " ")
                
                # Determine verdict
                if wins > losses:
                    verdict = "✓ Helps"
                elif losses > wins:
                    verdict = "✗ Hurts"
                else:
                    verdict = "~ Mixed"
                
                print(f"  {metric_short:<25}: Median {median_imp:+6.1f}% | Wins: {wins}, Losses: {losses}, Neutral: {neutral} | {verdict}")
        
        # Add conclusion
        print("\n  " + "-" * 76)
        print("  CONCLUSION: Enhanced prompts are CONTENT-DEPENDENT")
        print("    ✓ USE for: Dynamic natural motion (birds, animals, waving)")
        print("    ✗ AVOID for: Static content (portraits, landscapes) and fast action (jets)")
        print("  " + "-" * 76)
    
    print(f"\n\n{'='*80}")
    print(" SAVING CSV FILES")
    print("=" * 80 + "\n")
    
    save_all_csvs(df, output_dir)
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

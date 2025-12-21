"""
Experiment 08: Comprehensive Grid Search Analysis

Generates all analysis tables with win-rates across multiple metrics.
Outputs both console tables and CSV files.

Usage:
    python experiments/08_analyze_comprehensive.py
    python experiments/08_analyze_comprehensive.py --input path/to/results.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("outputs/06_grid_search_metrics/grid_search_results.json")
OUTPUT_DIR = Path("outputs/08_comprehensive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All metrics to analyze (lower is better for all)
METRICS = [
    "mean_mse",
    "mean_lpips",
    "mean_flow_magnitude",
    "flow_magnitude_variance",
    "mean_warp_error",
    "warp_error_variance",
    "flicker_index",
    "temporal_consistency_score"
]

# Short names for display
METRIC_SHORT_NAMES = {
    "mean_mse": "MSE",
    "mean_lpips": "LPIPS",
    "mean_flow_magnitude": "Flow Mag",
    "flow_magnitude_variance": "Flow Var",
    "mean_warp_error": "Warp Err",
    "warp_error_variance": "Warp Var",
    "flicker_index": "Flicker",
    "temporal_consistency_score": "Consistency"
}


# ============================================================
# Data Loading
# ============================================================

def load_results(json_path: Path) -> pd.DataFrame:
    """Load results JSON and convert to DataFrame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ============================================================
# CFG Analysis
# ============================================================

def analyze_cfg_sweep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze CFG sweep across all videos and metrics.
    
    Returns:
        - detailed_df: Per-video, per-metric optimal CFG values
        - summary_df: Win counts per CFG value per metric
        - trends_df: Trend analysis (higher/lower CFG better)
    """
    cfg_data = df[(df["num_inference_steps"] == 25) & (df["phase"] == "cfg_ablation")]
    videos = df["video_name"].unique()
    
    # Detailed results
    detailed_rows = []
    for video in sorted(videos):
        video_cfg = cfg_data[cfg_data["video_name"] == video]
        if video_cfg.empty:
            continue
        
        row = {"video": video}
        for metric in METRICS:
            best_idx = video_cfg[metric].idxmin()
            worst_idx = video_cfg[metric].idxmax()
            
            row[f"{metric}_best_cfg"] = video_cfg.loc[best_idx, "guidance_scale"]
            row[f"{metric}_best_val"] = video_cfg.loc[best_idx, metric]
            row[f"{metric}_worst_cfg"] = video_cfg.loc[worst_idx, "guidance_scale"]
            row[f"{metric}_worst_val"] = video_cfg.loc[worst_idx, metric]
            
            # Trend
            low_cfg = video_cfg[video_cfg["guidance_scale"] <= 6.0][metric].mean()
            high_cfg = video_cfg[video_cfg["guidance_scale"] >= 8.0][metric].mean()
            
            if high_cfg < low_cfg * 0.9:
                row[f"{metric}_trend"] = "Higher CFG better"
            elif low_cfg < high_cfg * 0.9:
                row[f"{metric}_trend"] = "Lower CFG better"
            else:
                row[f"{metric}_trend"] = "Mixed"
        
        detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Summary: win counts
    summary_rows = []
    for metric in METRICS:
        cfg_wins = {}
        for video in videos:
            video_cfg = cfg_data[cfg_data["video_name"] == video]
            if not video_cfg.empty:
                best_cfg = video_cfg.loc[video_cfg[metric].idxmin(), "guidance_scale"]
                cfg_wins[best_cfg] = cfg_wins.get(best_cfg, 0) + 1
        
        row = {"metric": metric}
        for cfg in [5.0, 6.0, 7.0, 7.5, 8.0, 9.0]:
            row[f"cfg_{cfg}"] = cfg_wins.get(cfg, 0)
        
        if cfg_wins:
            winner = max(cfg_wins.items(), key=lambda x: x[1])
            row["winner_cfg"] = winner[0]
            row["winner_count"] = winner[1]
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Trends summary
    trends_rows = []
    for metric in METRICS:
        trends = {"Higher CFG better": 0, "Lower CFG better": 0, "Mixed": 0}
        for video in videos:
            video_cfg = cfg_data[cfg_data["video_name"] == video]
            if video_cfg.empty:
                continue
            
            low_cfg = video_cfg[video_cfg["guidance_scale"] <= 6.0][metric].mean()
            high_cfg = video_cfg[video_cfg["guidance_scale"] >= 8.0][metric].mean()
            
            if high_cfg < low_cfg * 0.9:
                trends["Higher CFG better"] += 1
            elif low_cfg < high_cfg * 0.9:
                trends["Lower CFG better"] += 1
            else:
                trends["Mixed"] += 1
        
        trends_rows.append({
            "metric": metric,
            **trends,
            "dominant_trend": max(trends.items(), key=lambda x: x[1])[0]
        })
    
    trends_df = pd.DataFrame(trends_rows)
    
    return detailed_df, summary_df, trends_df


# ============================================================
# Steps Analysis
# ============================================================

def analyze_steps_sweep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze Steps sweep across all videos and metrics.
    
    Returns:
        - detailed_df: Per-video, per-metric optimal Steps values
        - summary_df: Win counts per Steps value per metric
        - trends_df: Trend analysis (more/fewer steps better)
    """
    steps_data = df[(df["guidance_scale"] == 7.5) & (df["phase"] == "steps_ablation")]
    videos = df["video_name"].unique()
    
    # Detailed results
    detailed_rows = []
    for video in sorted(videos):
        video_steps = steps_data[steps_data["video_name"] == video]
        if video_steps.empty:
            continue
        
        row = {"video": video}
        for metric in METRICS:
            best_idx = video_steps[metric].idxmin()
            worst_idx = video_steps[metric].idxmax()
            
            row[f"{metric}_best_steps"] = int(video_steps.loc[best_idx, "num_inference_steps"])
            row[f"{metric}_best_val"] = video_steps.loc[best_idx, metric]
            row[f"{metric}_worst_steps"] = int(video_steps.loc[worst_idx, "num_inference_steps"])
            row[f"{metric}_worst_val"] = video_steps.loc[worst_idx, metric]
            
            # Trend
            low_steps = video_steps[video_steps["num_inference_steps"] <= 20][metric].mean()
            high_steps = video_steps[video_steps["num_inference_steps"] >= 40][metric].mean()
            
            if high_steps < low_steps * 0.9:
                row[f"{metric}_trend"] = "More steps better"
            elif low_steps < high_steps * 0.9:
                row[f"{metric}_trend"] = "Fewer steps better"
            else:
                row[f"{metric}_trend"] = "Mixed"
        
        detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Summary: win counts
    summary_rows = []
    for metric in METRICS:
        steps_wins = {}
        for video in videos:
            video_steps = steps_data[steps_data["video_name"] == video]
            if not video_steps.empty:
                best_steps = int(video_steps.loc[video_steps[metric].idxmin(), "num_inference_steps"])
                steps_wins[best_steps] = steps_wins.get(best_steps, 0) + 1
        
        row = {"metric": metric}
        for steps in [15, 20, 25, 30, 40, 50]:
            row[f"steps_{steps}"] = steps_wins.get(steps, 0)
        
        if steps_wins:
            winner = max(steps_wins.items(), key=lambda x: x[1])
            row["winner_steps"] = winner[0]
            row["winner_count"] = winner[1]
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Trends summary
    trends_rows = []
    for metric in METRICS:
        trends = {"More steps better": 0, "Fewer steps better": 0, "Mixed": 0}
        for video in videos:
            video_steps = steps_data[steps_data["video_name"] == video]
            if video_steps.empty:
                continue
            
            low_steps = video_steps[video_steps["num_inference_steps"] <= 20][metric].mean()
            high_steps = video_steps[video_steps["num_inference_steps"] >= 40][metric].mean()
            
            if high_steps < low_steps * 0.9:
                trends["More steps better"] += 1
            elif low_steps < high_steps * 0.9:
                trends["Fewer steps better"] += 1
            else:
                trends["Mixed"] += 1
        
        trends_rows.append({
            "metric": metric,
            **trends,
            "dominant_trend": max(trends.items(), key=lambda x: x[1])[0]
        })
    
    trends_df = pd.DataFrame(trends_rows)
    
    return detailed_df, summary_df, trends_df


# ============================================================
# Prompt Analysis
# ============================================================

def analyze_prompt_impact(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze prompt engineering impact.
    
    Returns:
        - detailed_df: Per-video percentage change for each metric
        - summary_df: Average improvement, win/loss counts per metric
    """
    prompt_data = df[df["phase"] == "prompt_ablation"]
    videos = df["video_name"].unique()
    
    # Detailed results
    detailed_rows = []
    for video in sorted(videos):
        video_prompt = prompt_data[prompt_data["video_name"] == video]
        baseline = video_prompt[video_prompt["experiment_id"].str.contains("baseline")]
        enhanced = video_prompt[video_prompt["experiment_id"].str.contains("enhanced")]
        
        if baseline.empty or enhanced.empty:
            continue
        
        row = {"video": video}
        wins = 0
        losses = 0
        
        for metric in METRICS:
            b_val = baseline[metric].values[0]
            e_val = enhanced[metric].values[0]
            
            if b_val != 0:
                pct_change = (b_val - e_val) / b_val * 100
                row[f"{metric}_baseline"] = b_val
                row[f"{metric}_enhanced"] = e_val
                row[f"{metric}_change_pct"] = pct_change
                
                if pct_change > 5:
                    row[f"{metric}_verdict"] = "Improved"
                    wins += 1
                elif pct_change < -5:
                    row[f"{metric}_verdict"] = "Worse"
                    losses += 1
                else:
                    row[f"{metric}_verdict"] = "Neutral"
        
        row["total_wins"] = wins
        row["total_losses"] = losses
        row["overall_verdict"] = "Helps" if wins > losses else ("Hurts" if losses > wins else "Neutral")
        
        detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Summary
    summary_rows = []
    for metric in METRICS:
        improvements = []
        wins = 0
        losses = 0
        
        for video in videos:
            video_prompt = prompt_data[prompt_data["video_name"] == video]
            baseline = video_prompt[video_prompt["experiment_id"].str.contains("baseline")]
            enhanced = video_prompt[video_prompt["experiment_id"].str.contains("enhanced")]
            
            if baseline.empty or enhanced.empty:
                continue
            
            b_val = baseline[metric].values[0]
            e_val = enhanced[metric].values[0]
            
            if b_val != 0:
                pct_change = (b_val - e_val) / b_val * 100
                improvements.append(pct_change)
                
                if pct_change > 5:
                    wins += 1
                elif pct_change < -5:
                    losses += 1
        
        if improvements:
            summary_rows.append({
                "metric": metric,
                "avg_improvement_pct": np.mean(improvements),
                "std_improvement_pct": np.std(improvements),
                "wins": wins,
                "losses": losses,
                "neutral": len(improvements) - wins - losses,
                "verdict": "Helps" if wins > losses else ("Hurts" if losses > wins else "Mixed")
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    return detailed_df, summary_df


# ============================================================
# Metric Agreement Analysis
# ============================================================

def analyze_metric_agreement(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze how well metrics agree on optimal values.
    
    Returns:
        - cfg_agreement_df: Per-video CFG agreement
        - steps_agreement_df: Per-video Steps agreement
    """
    cfg_data = df[(df["num_inference_steps"] == 25) & (df["phase"] == "cfg_ablation")]
    steps_data = df[(df["guidance_scale"] == 7.5) & (df["phase"] == "steps_ablation")]
    videos = df["video_name"].unique()
    
    # CFG agreement
    cfg_rows = []
    for video in sorted(videos):
        video_cfg = cfg_data[cfg_data["video_name"] == video]
        if video_cfg.empty:
            continue
        
        row = {"video": video}
        optimal_cfgs = []
        
        for metric in METRICS:
            best_cfg = video_cfg.loc[video_cfg[metric].idxmin(), "guidance_scale"]
            row[f"{metric}_best_cfg"] = best_cfg
            optimal_cfgs.append(best_cfg)
        
        row["unique_values"] = len(set(optimal_cfgs))
        row["agreement_score"] = 1 - (len(set(optimal_cfgs)) - 1) / (len(METRICS) - 1)
        row["most_common_cfg"] = max(set(optimal_cfgs), key=optimal_cfgs.count)
        
        cfg_rows.append(row)
    
    cfg_agreement_df = pd.DataFrame(cfg_rows)
    
    # Steps agreement
    steps_rows = []
    for video in sorted(videos):
        video_steps = steps_data[steps_data["video_name"] == video]
        if video_steps.empty:
            continue
        
        row = {"video": video}
        optimal_steps = []
        
        for metric in METRICS:
            best_steps = int(video_steps.loc[video_steps[metric].idxmin(), "num_inference_steps"])
            row[f"{metric}_best_steps"] = best_steps
            optimal_steps.append(best_steps)
        
        row["unique_values"] = len(set(optimal_steps))
        row["agreement_score"] = 1 - (len(set(optimal_steps)) - 1) / (len(METRICS) - 1)
        row["most_common_steps"] = max(set(optimal_steps), key=optimal_steps.count)
        
        steps_rows.append(row)
    
    steps_agreement_df = pd.DataFrame(steps_rows)
    
    return cfg_agreement_df, steps_agreement_df


# ============================================================
# Recommendations Generation
# ============================================================

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate content-type specific recommendations."""
    cfg_data = df[(df["num_inference_steps"] == 25) & (df["phase"] == "cfg_ablation")]
    steps_data = df[(df["guidance_scale"] == 7.5) & (df["phase"] == "steps_ablation")]
    prompt_data = df[df["phase"] == "prompt_ablation"]
    videos = df["video_name"].unique()
    
    rows = []
    for video in sorted(videos):
        row = {"video": video}
        
        # Best CFG (by win count across metrics)
        video_cfg = cfg_data[cfg_data["video_name"] == video]
        if not video_cfg.empty:
            cfg_wins = {}
            for metric in METRICS:
                best_cfg = video_cfg.loc[video_cfg[metric].idxmin(), "guidance_scale"]
                cfg_wins[best_cfg] = cfg_wins.get(best_cfg, 0) + 1
            row["recommended_cfg"] = max(cfg_wins.items(), key=lambda x: x[1])[0]
            row["cfg_confidence"] = max(cfg_wins.values()) / len(METRICS)
        
        # Best Steps (by win count across metrics)
        video_steps = steps_data[steps_data["video_name"] == video]
        if not video_steps.empty:
            steps_wins = {}
            for metric in METRICS:
                best_steps = int(video_steps.loc[video_steps[metric].idxmin(), "num_inference_steps"])
                steps_wins[best_steps] = steps_wins.get(best_steps, 0) + 1
            row["recommended_steps"] = max(steps_wins.items(), key=lambda x: x[1])[0]
            row["steps_confidence"] = max(steps_wins.values()) / len(METRICS)
        
        # Prompt recommendation
        video_prompt = prompt_data[prompt_data["video_name"] == video]
        baseline = video_prompt[video_prompt["experiment_id"].str.contains("baseline")]
        enhanced = video_prompt[video_prompt["experiment_id"].str.contains("enhanced")]
        
        if not baseline.empty and not enhanced.empty:
            wins = 0
            losses = 0
            for metric in METRICS:
                b_val = baseline[metric].values[0]
                e_val = enhanced[metric].values[0]
                if b_val != 0:
                    pct = (b_val - e_val) / b_val * 100
                    if pct > 5:
                        wins += 1
                    elif pct < -5:
                        losses += 1
            
            row["prompt_wins"] = wins
            row["prompt_losses"] = losses
            row["use_enhanced_prompt"] = "Yes" if wins > losses else ("No" if losses > wins else "Optional")
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================
# Printing Functions
# ============================================================

def print_table(df: pd.DataFrame, title: str):
    """Print a formatted table."""
    print(f"\n{'='*100}")
    print(f" {title}")
    print(f"{'='*100}")
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print(df.to_string(index=False))


def save_all_tables(
    cfg_detailed, cfg_summary, cfg_trends,
    steps_detailed, steps_summary, steps_trends,
    prompt_detailed, prompt_summary,
    cfg_agreement, steps_agreement,
    recommendations,
    output_dir: Path
):
    """Save all tables to CSV files."""
    tables = {
        "cfg_detailed": cfg_detailed,
        "cfg_summary": cfg_summary,
        "cfg_trends": cfg_trends,
        "steps_detailed": steps_detailed,
        "steps_summary": steps_summary,
        "steps_trends": steps_trends,
        "prompt_detailed": prompt_detailed,
        "prompt_summary": prompt_summary,
        "cfg_agreement": cfg_agreement,
        "steps_agreement": steps_agreement,
        "recommendations": recommendations,
    }
    
    for name, df in tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Grid Search Analysis")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return
    
    print("Loading results...")
    df = load_results(input_path)
    print(f"Loaded {len(df)} experiments across {df['video_name'].nunique()} videos")
    
    # Run all analyses
    print("\nAnalyzing CFG sweep...")
    cfg_detailed, cfg_summary, cfg_trends = analyze_cfg_sweep(df)
    
    print("Analyzing Steps sweep...")
    steps_detailed, steps_summary, steps_trends = analyze_steps_sweep(df)
    
    print("Analyzing Prompt impact...")
    prompt_detailed, prompt_summary = analyze_prompt_impact(df)
    
    print("Analyzing Metric agreement...")
    cfg_agreement, steps_agreement = analyze_metric_agreement(df)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(df)
    
    # Print key tables
    print_table(cfg_summary, "CFG SWEEP: Win Counts Per Metric")
    print_table(cfg_trends, "CFG SWEEP: Trend Analysis")
    print_table(steps_summary, "STEPS SWEEP: Win Counts Per Metric")
    print_table(steps_trends, "STEPS SWEEP: Trend Analysis")
    print_table(prompt_summary, "PROMPT ENGINEERING: Impact Summary")
    print_table(cfg_agreement[["video", "unique_values", "agreement_score", "most_common_cfg"]], 
                "CFG: Metric Agreement Per Video")
    print_table(steps_agreement[["video", "unique_values", "agreement_score", "most_common_steps"]], 
                "STEPS: Metric Agreement Per Video")
    print_table(recommendations, "FINAL RECOMMENDATIONS")
    
    # Save all tables
    print(f"\n{'='*100}")
    print(" SAVING CSV FILES")
    print(f"{'='*100}\n")
    
    save_all_tables(
        cfg_detailed, cfg_summary, cfg_trends,
        steps_detailed, steps_summary, steps_trends,
        prompt_detailed, prompt_summary,
        cfg_agreement, steps_agreement,
        recommendations,
        output_dir
    )
    
    print(f"\n{'='*100}")
    print(" ANALYSIS COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

"""
Experiment 06: Measure All Ablation Results

Runs full measurement pipeline on all generated ablation experiments
and produces a comprehensive comparison report.

Usage:
    # First run the ablation experiments
    python experiments/05_hyperparameter_ablation.py
    
    # Then measure all results
    python experiments/06_measure_ablations.py
    
    # Or measure specific experiments
    python experiments/06_measure_ablations.py --filter birds_flying
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ABLATION_DIR = Path("outputs/05_ablation")
OUTPUT_DIR = Path("outputs/06_ablation_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Classes
# ============================================================

@dataclass
class FramePairMetrics:
    """Metrics between two consecutive frames."""
    frame_idx: int
    mse: float
    psnr: float
    lpips: Optional[float]
    flow_magnitude: Optional[float]
    warp_error: Optional[float]


@dataclass
class ExperimentMetrics:
    """Complete metrics for one ablation experiment."""
    experiment_id: str
    video_name: str
    hypothesis: str
    config: Dict
    
    # Aggregate metrics
    num_frames: int
    mse_mean: float
    mse_std: float
    mse_variance: float
    psnr_mean: float
    
    lpips_mean: Optional[float]
    lpips_std: Optional[float]
    
    flow_magnitude_mean: Optional[float]
    flow_variance: Optional[float]
    warp_error_mean: Optional[float]
    
    # Derived scores
    temporal_consistency_score: float
    flicker_index: float
    
    # Per-frame data
    frame_metrics: List[Dict]


# ============================================================
# Metric Computers
# ============================================================

class LPIPSComputer:
    """Lazy-loaded LPIPS metric."""
    
    def __init__(self):
        self.model = None
        self.available = False
    
    def load(self):
        if self.model is not None:
            return
        try:
            import lpips
            self.model = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
            self.model.eval()
            self.available = True
            print("  LPIPS loaded")
        except ImportError:
            warnings.warn("LPIPS not available")
            self.available = False
    
    def compute(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Optional[float]:
        if not self.available:
            return None
        f1 = (frame1.unsqueeze(0) * 2 - 1).to(DEVICE)
        f2 = (frame2.unsqueeze(0) * 2 - 1).to(DEVICE)
        with torch.no_grad():
            return self.model(f1, f2).item()
    
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()


class FlowComputer:
    """Optical flow computation."""
    
    def __init__(self):
        self.available = False
        self.cv2 = None
    
    def load(self):
        try:
            import cv2
            self.cv2 = cv2
            self.available = True
            print("  Optical flow loaded")
        except ImportError:
            warnings.warn("OpenCV not available")
            self.available = False
    
    def compute_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Optional[np.ndarray]:
        if not self.available:
            return None
        gray1 = (frame1.mean(dim=0) * 255).numpy().astype(np.uint8)
        gray2 = (frame2.mean(dim=0) * 255).numpy().astype(np.uint8)
        flow = self.cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        return flow
    
    def flow_magnitude(self, flow: np.ndarray) -> float:
        return float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
    
    def warp_and_compare(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                          flow: np.ndarray) -> float:
        """Warp frame1 by flow and compute MSE with frame2."""
        C, H, W = frame1.shape
        
        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        sample_x = grid_x + flow[..., 0]
        sample_y = grid_y + flow[..., 1]
        
        sample_x = 2 * sample_x / (W - 1) - 1
        sample_y = 2 * sample_y / (H - 1) - 1
        
        grid = torch.stack([
            torch.from_numpy(sample_x),
            torch.from_numpy(sample_y)
        ], dim=-1).unsqueeze(0)
        
        frame_batch = frame1.unsqueeze(0)
        warped = F.grid_sample(
            frame_batch, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0)
        
        return F.mse_loss(warped, frame2).item()


# ============================================================
# Frame Loading
# ============================================================

def load_frames(frames_dir: Path) -> torch.Tensor:
    """Load frames from directory."""
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    
    frames = []
    for f in frame_files:
        img = Image.open(f).convert('RGB')
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)
    
    return torch.stack(frames)


# ============================================================
# Core Measurement
# ============================================================

def compute_psnr(mse: float) -> float:
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_flicker_index(frames: torch.Tensor) -> float:
    """Second-order difference to detect oscillation."""
    if len(frames) < 3:
        return 0.0
    
    flicker_scores = []
    for t in range(len(frames) - 2):
        second_diff = frames[t] - 2 * frames[t+1] + frames[t+2]
        flicker_scores.append(second_diff.abs().mean().item())
    
    return float(np.mean(flicker_scores))


def compute_consistency_score(mse_values: List[float], 
                               lpips_values: Optional[List[float]]) -> float:
    """Composite temporal consistency score."""
    mse_variance = np.var(mse_values)
    mse_mean = np.mean(mse_values)
    
    if lpips_values and len(lpips_values) > 0:
        lpips_mean = np.mean(lpips_values)
        lpips_variance = np.var(lpips_values)
        score = (mse_variance * 1000) + (mse_mean * 100) + (lpips_mean * 50) + (lpips_variance * 500)
    else:
        score = (mse_variance * 1000) + (mse_mean * 100)
    
    return float(score)


def measure_experiment(experiment_dir: Path, 
                        lpips_computer: LPIPSComputer,
                        flow_computer: FlowComputer) -> Optional[ExperimentMetrics]:
    """Measure a single ablation experiment."""
    
    frames_dir = experiment_dir / "frames"
    config_path = experiment_dir / "config.json"
    
    if not frames_dir.exists():
        print(f"  Skipping {experiment_dir.name}: no frames directory")
        return None
    
    if not config_path.exists():
        print(f"  Skipping {experiment_dir.name}: no config.json")
        return None
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load frames
    frames = load_frames(frames_dir)
    num_frames = len(frames)
    
    # Compute per-frame metrics
    frame_metrics = []
    mse_values = []
    lpips_values = []
    flow_magnitudes = []
    warp_errors = []
    
    for i in range(num_frames - 1):
        f1, f2 = frames[i], frames[i + 1]
        
        # MSE
        mse = F.mse_loss(f1, f2).item()
        mse_values.append(mse)
        
        # PSNR
        psnr = compute_psnr(mse)
        
        # LPIPS
        lpips_val = lpips_computer.compute(f1, f2)
        if lpips_val is not None:
            lpips_values.append(lpips_val)
        
        # Flow
        flow_mag = None
        warp_err = None
        if flow_computer.available:
            flow = flow_computer.compute_flow(f1, f2)
            if flow is not None:
                flow_mag = flow_computer.flow_magnitude(flow)
                flow_magnitudes.append(flow_mag)
                warp_err = flow_computer.warp_and_compare(f1, f2, flow)
                warp_errors.append(warp_err)
        
        frame_metrics.append({
            "frame_idx": i,
            "mse": mse,
            "psnr": psnr,
            "lpips": lpips_val,
            "flow_magnitude": flow_mag,
            "warp_error": warp_err,
        })
    
    # Aggregate metrics
    mse_mean = float(np.mean(mse_values))
    mse_std = float(np.std(mse_values))
    mse_variance = float(np.var(mse_values))
    psnr_mean = float(np.mean([fm["psnr"] for fm in frame_metrics]))
    
    lpips_mean = float(np.mean(lpips_values)) if lpips_values else None
    lpips_std = float(np.std(lpips_values)) if lpips_values else None
    
    flow_mean = float(np.mean(flow_magnitudes)) if flow_magnitudes else None
    flow_var = float(np.var(flow_magnitudes)) if flow_magnitudes else None
    warp_mean = float(np.mean(warp_errors)) if warp_errors else None
    
    # Derived scores
    consistency_score = compute_consistency_score(mse_values, lpips_values if lpips_values else None)
    flicker_index = compute_flicker_index(frames)
    
    return ExperimentMetrics(
        experiment_id=config.get("experiment_id", experiment_dir.name),
        video_name=config.get("video_name", "unknown"),
        hypothesis=config.get("hypothesis", "unknown"),
        config=config,
        num_frames=num_frames,
        mse_mean=mse_mean,
        mse_std=mse_std,
        mse_variance=mse_variance,
        psnr_mean=psnr_mean,
        lpips_mean=lpips_mean,
        lpips_std=lpips_std,
        flow_magnitude_mean=flow_mean,
        flow_variance=flow_var,
        warp_error_mean=warp_mean,
        temporal_consistency_score=consistency_score,
        flicker_index=flicker_index,
        frame_metrics=frame_metrics,
    )


# ============================================================
# Reporting
# ============================================================

def generate_comparison_report(all_metrics: List[ExperimentMetrics]) -> str:
    """Generate comprehensive comparison report."""
    
    lines = []
    lines.append("=" * 100)
    lines.append("ABLATION STUDY: COMPREHENSIVE METRICS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 100)
    lines.append("")
    
    # Group by video
    videos = {}
    for m in all_metrics:
        if m.video_name not in videos:
            videos[m.video_name] = []
        videos[m.video_name].append(m)
    
    # Summary table
    lines.append("SUMMARY TABLE")
    lines.append("-" * 100)
    header = (f"{'Experiment':<40} {'MSE Mean':<10} {'MSE Std':<10} {'LPIPS':<10} "
              f"{'Flow Var':<10} {'Consistency':<12} {'Δ%':<8}")
    lines.append(header)
    lines.append("-" * 100)
    
    improvement_stats = {}
    
    for video_name in sorted(videos.keys()):
        video_results = videos[video_name]
        
        # Find baseline
        baseline = next((m for m in video_results if m.hypothesis == "baseline"), None)
        baseline_score = baseline.temporal_consistency_score if baseline else None
        
        for m in sorted(video_results, key=lambda x: x.hypothesis):
            mse_str = f"{m.mse_mean:.5f}"
            mse_std_str = f"{m.mse_std:.5f}"
            lpips_str = f"{m.lpips_mean:.4f}" if m.lpips_mean else "N/A"
            flow_str = f"{m.flow_variance:.4f}" if m.flow_variance else "N/A"
            cons_str = f"{m.temporal_consistency_score:.4f}"
            
            # Compute improvement
            delta_str = ""
            if baseline_score and m.hypothesis != "baseline":
                delta = ((baseline_score - m.temporal_consistency_score) / baseline_score) * 100
                delta_str = f"{delta:+.1f}%"
                
                # Track improvements
                if m.hypothesis not in improvement_stats:
                    improvement_stats[m.hypothesis] = []
                improvement_stats[m.hypothesis].append(delta)
            
            lines.append(f"{m.experiment_id:<40} {mse_str:<10} {mse_std_str:<10} "
                        f"{lpips_str:<10} {flow_str:<10} {cons_str:<12} {delta_str:<8}")
        
        lines.append("")
    
    # Improvement summary by hypothesis
    lines.append("")
    lines.append("IMPROVEMENT SUMMARY BY INTERVENTION TYPE")
    lines.append("-" * 60)
    lines.append(f"{'Hypothesis':<20} {'Avg Improvement':<20} {'Best':<15} {'Worst':<15}")
    lines.append("-" * 60)
    
    for hyp in ["cfg_only", "steps_only", "prompt_only", "recommended"]:
        if hyp in improvement_stats and improvement_stats[hyp]:
            deltas = improvement_stats[hyp]
            avg = np.mean(deltas)
            best = max(deltas)
            worst = min(deltas)
            lines.append(f"{hyp:<20} {avg:+.1f}%{'':<15} {best:+.1f}%{'':<10} {worst:+.1f}%")
    
    # Per-video analysis
    lines.append("")
    lines.append("")
    lines.append("DETAILED PER-VIDEO ANALYSIS")
    lines.append("=" * 100)
    
    for video_name in sorted(videos.keys()):
        video_results = videos[video_name]
        baseline = next((m for m in video_results if m.hypothesis == "baseline"), None)
        
        lines.append("")
        lines.append(f"Video: {video_name}")
        lines.append("-" * 50)
        
        if baseline:
            lines.append(f"  Baseline consistency score: {baseline.temporal_consistency_score:.4f}")
        
        # Find best intervention
        non_baseline = [m for m in video_results if m.hypothesis != "baseline"]
        if non_baseline and baseline:
            best = min(non_baseline, key=lambda x: x.temporal_consistency_score)
            improvement = ((baseline.temporal_consistency_score - best.temporal_consistency_score) 
                          / baseline.temporal_consistency_score * 100)
            lines.append(f"  Best intervention: {best.hypothesis}")
            lines.append(f"  Best consistency score: {best.temporal_consistency_score:.4f} ({improvement:+.1f}%)")
            
            # What changed
            lines.append(f"  Changes applied:")
            if best.config.get("guidance_scale") != baseline.config.get("guidance_scale"):
                lines.append(f"    - CFG: {baseline.config.get('guidance_scale')} → {best.config.get('guidance_scale')}")
            if best.config.get("num_inference_steps") != baseline.config.get("num_inference_steps"):
                lines.append(f"    - Steps: {baseline.config.get('num_inference_steps')} → {best.config.get('num_inference_steps')}")
    
    # Conclusions
    lines.append("")
    lines.append("")
    lines.append("CONCLUSIONS")
    lines.append("=" * 100)
    
    # Find which hypothesis works best overall
    if improvement_stats:
        best_hyp = max(improvement_stats.keys(), 
                       key=lambda h: np.mean(improvement_stats[h]) if improvement_stats[h] else -999)
        best_avg = np.mean(improvement_stats[best_hyp])
        
        lines.append(f"1. Most effective intervention: {best_hyp} (avg {best_avg:+.1f}% improvement)")
        
        if "recommended" in improvement_stats:
            rec_avg = np.mean(improvement_stats["recommended"])
            lines.append(f"2. Combined 'recommended' settings: avg {rec_avg:+.1f}% improvement")
        
        if "cfg_only" in improvement_stats:
            cfg_avg = np.mean(improvement_stats["cfg_only"])
            lines.append(f"3. CFG reduction alone: avg {cfg_avg:+.1f}% improvement")
    
    lines.append("")
    lines.append("Note: Positive percentages indicate improvement (lower consistency score = better)")
    
    return "\n".join(lines)


def save_metrics_json(all_metrics: List[ExperimentMetrics], output_path: Path):
    """Save all metrics to JSON."""
    data = []
    for m in all_metrics:
        entry = asdict(m)
        data.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Measure all ablation experiments")
    parser.add_argument("--input", type=str, default=str(ABLATION_DIR),
                        help="Directory containing ablation experiments")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only measure experiments matching this string")
    parser.add_argument("--no-lpips", action="store_true",
                        help="Skip LPIPS computation")
    parser.add_argument("--no-flow", action="store_true",
                        help="Skip optical flow computation")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Run experiments/05_hyperparameter_ablation.py first")
        return
    
    # Find all experiment directories
    experiment_dirs = [d for d in input_dir.iterdir() 
                       if d.is_dir() and (d / "frames").exists()]
    
    if args.filter:
        experiment_dirs = [d for d in experiment_dirs if args.filter in d.name]
    
    print(f"Found {len(experiment_dirs)} experiments to measure")
    
    if not experiment_dirs:
        print("No experiments found!")
        return
    
    # Initialize metric computers
    lpips_computer = LPIPSComputer()
    flow_computer = FlowComputer()
    
    if not args.no_lpips:
        lpips_computer.load()
    if not args.no_flow:
        flow_computer.load()
    
    # Measure all experiments
    all_metrics = []
    
    for i, exp_dir in enumerate(sorted(experiment_dirs)):
        print(f"\n[{i+1}/{len(experiment_dirs)}] Measuring: {exp_dir.name}")
        
        metrics = measure_experiment(exp_dir, lpips_computer, flow_computer)
        
        if metrics:
            all_metrics.append(metrics)
            
            # Save individual metrics
            individual_path = OUTPUT_DIR / f"{metrics.experiment_id}_metrics.json"
            with open(individual_path, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
    
    # Cleanup LPIPS to free memory
    lpips_computer.cleanup()
    
    print(f"\n{'='*60}")
    print(f"Measured {len(all_metrics)} experiments")
    print(f"{'='*60}")
    
    # Generate and save comparison report
    report = generate_comparison_report(all_metrics)
    print("\n" + report)
    
    report_path = OUTPUT_DIR / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save all metrics to JSON
    all_metrics_path = OUTPUT_DIR / "all_metrics.json"
    save_metrics_json(all_metrics, all_metrics_path)
    print(f"All metrics saved to: {all_metrics_path}")
    
    print("\nMeasurement complete!")


if __name__ == "__main__":
    main()
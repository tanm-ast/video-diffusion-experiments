"""
Experiment 06: Measure Grid Search Results

Measures all experiments from the grid search.
Refactored from 05_measurement_pipeline.py structure.

Usage:
    python experiments/06_measure_grid_search.py
    python experiments/06_measure_grid_search.py --filter portrait
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

import lpips
import cv2

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR = Path("outputs/05_grid_search")
OUTPUT_DIR = Path("outputs/06_grid_search_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Classes (from 05_measurement_pipeline.py)
# ============================================================

@dataclass
class FramePairMetrics:
    """Metrics between two consecutive frames."""
    frame_idx: int
    mse: float
    psnr: float
    lpips: float
    flow_magnitude_mean: float
    flow_magnitude_std: float
    warp_error: float


@dataclass 
class VideoMetrics:
    """Aggregate metrics for a video."""
    video_name: str
    experiment_id: str
    num_frames: int
    
    # Config info
    guidance_scale: float
    num_inference_steps: int
    phase: str
    
    # Per-frame metrics
    frame_metrics: List[FramePairMetrics]
    
    # Aggregate statistics - MSE
    mean_mse: float
    std_mse: float
    
    # Aggregate statistics - PSNR
    mean_psnr: float
    
    # Aggregate statistics - LPIPS
    mean_lpips: float
    std_lpips: float
    
    # Aggregate statistics - Optical Flow
    mean_flow_magnitude: float
    flow_magnitude_variance: float
    
    # Aggregate statistics - Warp Error
    mean_warp_error: float
    warp_error_variance: float
    
    # Temporal consistency (composite score, lower = better)
    temporal_consistency_score: float
    
    # Flicker index (second-order temporal difference, requires 3 frames per measurement)
    # Computed at video level, not per frame pair
    flicker_index: float

# ============================================================
# Frame Loading
# ============================================================

def load_frames(frame_dir: Path) -> torch.Tensor:
    """Load frames from a directory. Returns [F, C, H, W] in [0, 1]."""
    frame_files = sorted(frame_dir.glob('*.png'))
    if not frame_files:
        frame_files = sorted(frame_dir.glob('*.jpg'))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frame_dir}")
    
    frames = []
    for f in frame_files:
        img = Image.open(f).convert('RGB')
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)
    
    return torch.stack(frames, dim=0)





# ============================================================
# Metric Computers (from 05_measurement_pipeline.py)
# ============================================================

class LPIPSMetric:
    """
    Wrapper for LPIPS perceptual similarity metric.
    Requires lpips package.
    """
    
    def __init__(self):
        self.model = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
        self.model.eval()
    
    def compute(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """
        Compute LPIPS between two frames.
        Converts inputs in [0,1] range to [-1, 1] required by LPIPS.
        
        Args:
            frame1, frame2: [C, H, W] tensors in [0, 1]
        
        Returns:
            LPIPS distance (lower = more similar)
        """
        f1 = (frame1.unsqueeze(0) * 2 - 1).to(DEVICE)
        f2 = (frame2.unsqueeze(0) * 2 - 1).to(DEVICE)
        
        with torch.no_grad():
            distance = self.model(f1, f2)
        
        return distance.item()
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()


class OpticalFlowEstimator:
    """
    Optical flow estimation for motion analysis.
    Uses OpenCV's Farneback method.
    """
    
    def compute_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> np.ndarray:
        """
        Compute optical flow from frame1 to frame2.
        
        Args:
            frame1, frame2: [C, H, W] tensors in [0, 1]
        
        Returns:
            Flow field [H, W, 2] (dx, dy per pixel)
        """
        gray1 = (frame1.mean(dim=0) * 255).numpy().astype(np.uint8)
        gray2 = (frame2.mean(dim=0) * 255).numpy().astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    def compute_flow_stats(self, flow: np.ndarray) -> dict:
        """Compute statistics from flow field."""
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return {
            'magnitude_mean': float(magnitude.mean()),
            'magnitude_std': float(magnitude.std()),
            'magnitude_max': float(magnitude.max()),
            'magnitude_median': float(np.median(magnitude)),
        }





# ============================================================
# Basic Metrics
# ============================================================

def compute_mse(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """Mean Squared Error between two frames."""
    return F.mse_loss(frame1, frame2).item()


def compute_psnr(mse: float) -> float:
    """Peak Signal-to-Noise Ratio from MSE."""
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_flicker_index(frames: torch.Tensor) -> float:
    """
    Second-order temporal difference to detect oscillation.
    Flicker Index = temporal mean of spatial mean of |I_t - 2*I_{t+1} + I_{t+2}|
    """
    if len(frames) < 3:
        return 0.0
    
    flicker_scores = []
    for t in range(len(frames) - 2):
        second_diff = frames[t] - 2 * frames[t+1] + frames[t+2]
        spatial_mean = second_diff.abs().mean().item()
        flicker_scores.append(spatial_mean)
    
    return float(np.mean(flicker_scores))


def compute_temporal_consistency_score(frame_metrics: List[FramePairMetrics]) -> float:
    """
    Composite temporal consistency score (lower = better).
    Combines MSE variance and LPIPS for overall consistency measure.
    """
    mse_values = [m.mse for m in frame_metrics]
    lpips_values = [m.lpips for m in frame_metrics]
    
    mse_variance = float(np.var(mse_values))
    mse_mean = float(np.mean(mse_values))
    lpips_mean = float(np.mean(lpips_values))
    lpips_variance = float(np.var(lpips_values))
    
    score = (mse_variance * 1000) + (mse_mean * 100) + (lpips_mean * 50) + (lpips_variance * 500)
    return score


# ============================================================
# Warping
# ============================================================

def warp_frame(frame: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
    """Warp frame using backward warping with flow field."""
    C, H, W = frame.shape
    
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    sample_x = grid_x + flow[..., 0]
    sample_y = grid_y + flow[..., 1]
    
    # Normalize to [-1, 1] for grid_sample
    sample_x = 2 * sample_x / (W - 1) - 1
    sample_y = 2 * sample_y / (H - 1) - 1
    
    grid = torch.stack([
        torch.from_numpy(sample_x),
        torch.from_numpy(sample_y)
    ], dim=-1).unsqueeze(0)
    
    frame_batch = frame.unsqueeze(0)
    warped = F.grid_sample(
        frame_batch, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0)
    
    return warped


# ============================================================
# Main Measurement Function
# ============================================================

def measure_video(
    frames: torch.Tensor,
    video_name: str,
    experiment_id: str,
    config: dict,
    lpips_metric: LPIPSMetric,
    flow_estimator: OpticalFlowEstimator
) -> VideoMetrics:
    """
    Measure temporal consistency metrics for a video.
    
    Args:
        frames: [F, C, H, W] tensor in [0, 1]
        video_name: Name of the video
        experiment_id: Experiment identifier
        config: Experiment configuration dict
        lpips_metric: LPIPS metric computer
        flow_estimator: Optical flow estimator
    
    Returns:
        VideoMetrics with all measurements
    """
    F_count = len(frames)
    frame_metrics = []
    all_flow_magnitudes = []
    all_warp_errors = []
    
    print(f"  Measuring {F_count} frames...")
    
    for i in range(F_count - 1):
        frame1, frame2 = frames[i], frames[i + 1]
        
        # Basic metrics
        mse = compute_mse(frame1, frame2)
        psnr = compute_psnr(mse)
        
        # Perceptual metric
        lpips_val = lpips_metric.compute(frame1, frame2)
        
        # Flow metrics
        flow = flow_estimator.compute_flow(frame1, frame2)
        flow_stats = flow_estimator.compute_flow_stats(flow)
        all_flow_magnitudes.append(flow_stats['magnitude_mean'])
        
        # Warp error
        warped = warp_frame(frame1, flow)
        warp_error = compute_mse(warped, frame2)
        all_warp_errors.append(warp_error)
        
        metrics = FramePairMetrics(
            frame_idx=i,
            mse=mse,
            psnr=psnr,
            lpips=lpips_val,
            flow_magnitude_mean=flow_stats['magnitude_mean'],
            flow_magnitude_std=flow_stats['magnitude_std'],
            warp_error=warp_error
        )
        frame_metrics.append(metrics)
    
    # Aggregate statistics
    mse_values = [m.mse for m in frame_metrics]
    psnr_values = [m.psnr for m in frame_metrics]
    lpips_values = [m.lpips for m in frame_metrics]
    
    # Temporal consistency score
    consistency_score = compute_temporal_consistency_score(frame_metrics)
    
    # Flicker index (requires frame triplets, computed at video level)
    flicker_idx = compute_flicker_index(frames)
    
    video_metrics = VideoMetrics(
        video_name=video_name,
        experiment_id=experiment_id,
        num_frames=F_count,
        guidance_scale=config.get("guidance_scale", 0),
        num_inference_steps=config.get("num_inference_steps", 0),
        phase=config.get("phase", "unknown"),
        frame_metrics=frame_metrics,
        # MSE aggregates
        mean_mse=float(np.mean(mse_values)),
        std_mse=float(np.std(mse_values)),
        # PSNR aggregate
        mean_psnr=float(np.mean(psnr_values)),
        # LPIPS aggregates
        mean_lpips=float(np.mean(lpips_values)),
        std_lpips=float(np.std(lpips_values)),
        # Flow aggregates
        mean_flow_magnitude=float(np.mean(all_flow_magnitudes)),
        flow_magnitude_variance=float(np.var(all_flow_magnitudes)),
        # Warp error aggregates
        mean_warp_error=float(np.mean(all_warp_errors)),
        warp_error_variance=float(np.var(all_warp_errors)),
        # Composite scores
        temporal_consistency_score=consistency_score,
        flicker_index=flicker_idx,
    )
    
    return video_metrics


# ============================================================
# Serialization
# ============================================================

def save_metrics(metrics: VideoMetrics, output_path: Path):
    """Save metrics to JSON."""
    data = {
        'video_name': metrics.video_name,
        'experiment_id': metrics.experiment_id,
        'num_frames': metrics.num_frames,
        'guidance_scale': metrics.guidance_scale,
        'num_inference_steps': metrics.num_inference_steps,
        'phase': metrics.phase,
        # MSE
        'mean_mse': metrics.mean_mse,
        'std_mse': metrics.std_mse,
        # PSNR
        'mean_psnr': metrics.mean_psnr,
        # LPIPS
        'mean_lpips': metrics.mean_lpips,
        'std_lpips': metrics.std_lpips,
        # Flow
        'mean_flow_magnitude': metrics.mean_flow_magnitude,
        'flow_magnitude_variance': metrics.flow_magnitude_variance,
        # Warp error
        'mean_warp_error': metrics.mean_warp_error,
        'warp_error_variance': metrics.warp_error_variance,
        # Composite
        'temporal_consistency_score': metrics.temporal_consistency_score,
        'flicker_index': metrics.flicker_index,
        # Per-frame
        'frame_metrics': [asdict(fm) for fm in metrics.frame_metrics]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def save_summary(all_metrics: List[VideoMetrics], output_path: Path):
    """Save summary of all metrics for analysis."""
    summary = []
    for m in all_metrics:
        summary.append({
            "experiment_id": m.experiment_id,
            "video_name": m.video_name,
            "guidance_scale": m.guidance_scale,
            "num_inference_steps": m.num_inference_steps,
            "phase": m.phase,
            # MSE
            "mean_mse": m.mean_mse,
            "std_mse": m.std_mse,
            # LPIPS
            "mean_lpips": m.mean_lpips,
            "std_lpips": m.std_lpips,
            # Flow
            "mean_flow_magnitude": m.mean_flow_magnitude,
            "flow_magnitude_variance": m.flow_magnitude_variance,
            # Warp error
            "mean_warp_error": m.mean_warp_error,
            "warp_error_variance": m.warp_error_variance,
            # Composite
            "temporal_consistency_score": m.temporal_consistency_score,
            "flicker_index": m.flicker_index,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Measure grid search experiments")
    parser.add_argument("--input", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--filter", type=str, default=None,
                        help="Only measure experiments matching this string")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        print("Run experiments/05_grid_search_ablation.py first")
        return
    
    # Find experiments
    experiment_dirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "frames").exists()
    ]
    
    if args.filter:
        experiment_dirs = [d for d in experiment_dirs if args.filter in d.name]
    
    print(f"Found {len(experiment_dirs)} experiments to measure")
    
    if not experiment_dirs:
        return
    
    # Initialize metric computers
    print("\nInitializing metrics...")
    lpips_metric = LPIPSMetric()
    flow_estimator = OpticalFlowEstimator()
    
    # Measure all experiments
    all_metrics = []
    
    for i, exp_dir in enumerate(sorted(experiment_dirs)):
        print(f"\n[{i+1}/{len(experiment_dirs)}] {exp_dir.name}")
        
        # Load config
        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Load frames
        frames_dir = exp_dir / "frames"
        frames = load_frames(frames_dir)
        
        # Measure
        metrics = measure_video(
            frames=frames,
            video_name=config["video_name"],
            experiment_id=config["experiment_id"],
            config=config,
            lpips_metric=lpips_metric,
            flow_estimator=flow_estimator,
        )
        
        all_metrics.append(metrics)
        
        # Save individual metrics
        individual_path = output_dir / f"{metrics.experiment_id}_metrics.json"
        save_metrics(metrics, individual_path)
    
    # Cleanup
    lpips_metric.cleanup()
    
    # Save summary
    summary_path = output_dir / "grid_search_results.json"
    save_summary(all_metrics, summary_path)
    
    print(f"\n{'='*60}")
    print(f"Measurement complete!")
    print(f"{'='*60}")
    print(f"Results saved: {summary_path}")
    print(f"Total experiments measured: {len(all_metrics)}")
    print(f"\nNext step: python experiments/07_analyze_grid_search.py")


if __name__ == "__main__":
    main()

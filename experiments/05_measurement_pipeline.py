"""
Experiment 05: Temporal Consistency Measurement Pipeline

Quantify flickering and temporal inconsistency in generated videos.

Metrics:
1. Inter-frame LPIPS (perceptual difference between consecutive frames)
2. Pixel MSE between consecutive frames
3. Optical flow magnitude variance (motion smoothness)
4. Warping error (flow-warped frame vs actual frame)

Usage:
    python experiments/05_measurement_pipeline.py --input outputs/01_baseline/corgi_beach_frames
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import warnings
import lpips
import cv2

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/05_measurements")
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
    flow_magnitude_mean: Optional[float]
    flow_magnitude_std: Optional[float]
    warp_error: Optional[float]


@dataclass 
class VideoMetrics:
    """Aggregate metrics for a video."""
    video_name: str
    num_frames: int
    
    # Per-frame metrics (lists)
    frame_metrics: List[FramePairMetrics]
    
    # Aggregate statistics
    mean_mse: float
    std_mse: float
    mean_psnr: float
    mean_lpips: Optional[float]
    std_lpips: Optional[float]
    
    # Temporal consistency score (lower = more consistent)
    temporal_consistency_score: float
    
    # Flow statistics
    mean_flow_magnitude: Optional[float]
    flow_magnitude_variance: Optional[float]


# ============================================================
# Frame Loading
# ============================================================

def load_frames(frame_dir: Path) -> torch.Tensor:
    """
    Load frames from a directory.
    
    Returns:
        Tensor of shape [F, C, H, W] in range [0, 1]
    """
    frame_dir = Path(frame_dir)
    
    # Find frame files
    extensions = ['*.png', '*.jpg', '*.jpeg']
    frame_files = []
    for ext in extensions:
        frame_files.extend(sorted(frame_dir.glob(ext)))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frame_dir}")
    
    frames = []
    for f in frame_files:
        img = Image.open(f).convert('RGB')
        # Convert to tensor [C, H, W] in [0, 1]
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)
    
    # Stack to [F, C, H, W]
    frames = torch.stack(frames, dim=0)
    print(f"Loaded {len(frames)} frames of shape {frames.shape[1:]}")
    
    return frames


# ============================================================
# Basic Metrics (No Dependencies)
# ============================================================

def compute_mse(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """Mean Squared Error between two frames."""
    return F.mse_loss(frame1, frame2).item()


def compute_psnr(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = compute_mse(frame1, frame2)
    if mse < 1e-10:
        return 100.0  # Essentially identical
    '''
    PSNR = 10 * log10(MAX_I^2 / MSE)
    For images in [0, 1], MAX_I = 1
    '''
    return (10 * np.log10(1.0 / mse))


def compute_pixel_diff_stats(frame1: torch.Tensor, frame2: torch.Tensor) -> dict:
    """Compute various pixel difference statistics."""
    diff = (frame1 - frame2).abs()
    return {
        'mean_abs_diff': diff.mean().item(),
        'max_abs_diff': diff.max().item(),
        'std_abs_diff': diff.std().item(),
    }


# ============================================================
# LPIPS (Perceptual Metric)
# ============================================================

class LPIPSMetric:
    """
    Wrapper for LPIPS perceptual similarity metric.
    Requires lpips package.
    """
    
    def __init__(self):
        self.model = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
        self.model.eval()
    
    
    def compute(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Optional[float]:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) between two frames.
        Compares extracted features from a pretrained network (Default: AlexNet).
        Converts inputs in [0,1] range  to [-1, 1] required by LPIPS.
        Args:
            frame1, frame2: [C, H, W] tensors in [0, 1]
        
        Returns:
            LPIPS distance (lower = more similar)
        """
    
        
        # LPIPS expects [B, C, H, W] in [-1, 1]
        f1 = (frame1.unsqueeze(0) * 2 - 1).to(DEVICE)
        f2 = (frame2.unsqueeze(0) * 2 - 1).to(DEVICE)
        
        with torch.no_grad():
            distance = self.model(f1, f2)
        
        return distance.item()


# ============================================================
# Optical Flow (Motion Analysis)
# ============================================================

class OpticalFlowEstimator:
    """
    Optical flow estimation for motion analysis.
    Uses OpenCV's Farneback method (CPU, always available).
    """
    
    def __init__(self):
        self.cv2 = cv2
    
    def compute_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Optional[np.ndarray]:
        """
        Compute optical flow from frame1 to frame2.
        
        Args:
            frame1, frame2: [C, H, W] tensors in [0, 1]
        
        Returns:
            Flow field [H, W, 2] (dx, dy per pixel)
        """
        
        # Convert to grayscale numpy [H, W] in [0, 255]
        gray1 = (frame1.mean(dim=0) * 255).numpy().astype(np.uint8)
        gray2 = (frame2.mean(dim=0) * 255).numpy().astype(np.uint8)
        
        # Farneback optical flow
        flow = self.cv2.calcOpticalFlowFarneback(
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
        """
        Compute statistics from flow field.
        """
        # Flow magnitude at each pixel
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return {
            'magnitude_mean': float(magnitude.mean()),
            'magnitude_std': float(magnitude.std()),
            'magnitude_max': float(magnitude.max()),
            'magnitude_median': float(np.median(magnitude)),
        }
    
    def warp_frame(self, frame: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
        """
        Warp frame according to flow field.
        
        Args:
            frame: [C, H, W] tensor
            flow: [H, W, 2] numpy array
        
        Returns:
            Warped frame [C, H, W]
        """
        
        
        C, H, W = frame.shape
        
        # Create sampling grid
        # flow[y, x] = (dx, dy) means pixel at (x, y) in frame1 
        # moves TO position (x+dx, y+dy) in frame2
        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        
        # For backward warping (creating warped frame1 to match frame2):
        # We approximate by sampling: output[y,x] ← input[y + dy, x + dx]
        # This uses forward flow as an approximation of inverse flow
        sample_x = grid_x + flow[..., 0]
        sample_y = grid_y + flow[..., 1]
        
        # Normalize to [-1, 1] for grid_sample
        sample_x = 2 * sample_x / (W - 1) - 1
        sample_y = 2 * sample_y / (H - 1) - 1
        
        '''
        The grid contains sampling coordinates: "for each output pixel, where should I read from in the input?
        '''
        # Create grid [1, H, W, 2]
        grid = torch.stack([
            torch.from_numpy(sample_x),
            torch.from_numpy(sample_y)
        ], dim=-1).unsqueeze(0)
        
        # Warp
        '''
        grid_sample performs spatial sampling from an input tensor at arbitrary continuous locations.
        For each output position (i, j):

        1) Look up the sampling coordinates: (x, y) = grid[b, i, j, :]
        2) These coordinates are in normalized [-1, 1] space
        3) Convert back to pixel coordinates
        4) Sample from input at that location using interpolation
        5) Write to output[b, :, i, j]
        '''
        frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
        warped = F.grid_sample(
            frame_batch, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped.squeeze(0)


# ============================================================
# Temporal Consistency Metrics
# ============================================================

def compute_temporal_consistency_score(frame_metrics: List[FramePairMetrics]) -> float:
    """
    Compute overall temporal consistency score.
    
    Higher values = more inconsistency (flickering)
    
    Combines:
    - Variance in inter-frame differences (should be stable)
    - Mean perceptual difference (LPIPS)
    """
    mse_values = [m.mse for m in frame_metrics]
    
    # Variance in MSE indicates inconsistent frame-to-frame changes
    mse_variance = np.var(mse_values)
    
    # Mean MSE indicates overall difference magnitude
    mse_mean = np.mean(mse_values)
    
    # If LPIPS available, factor it in
    lpips_values = [m.lpips for m in frame_metrics if m.lpips is not None]
    if lpips_values:
        lpips_mean = np.mean(lpips_values)
        lpips_variance = np.var(lpips_values)
        # Weighted combination
        # The weights are heuristic and can be tuned
        # mse_variance: ~0.00001 to 0.001 (very small)
        # mse_mean: ~0.001 to 0.1
        # lpips_mean: ~0.05 to 0.5
        # lpips_variance: ~0.0001 to 0.01
        print("Displaying inter-frame inconsistency components:")
        print(f"  MSE Variance: {mse_variance:.6f}")
        print(f"  MSE Mean:     {mse_mean:.6f}")
        print(f"  LPIPS Mean:   {lpips_mean:.6f}")  
        print(f"  LPIPS Variance: {lpips_variance:.6f}")
        print("Expected Ranges: MSE Var(~0.00001-0.001), MSE Mean(~0.001-0.1), LPIPS Mean(~0.05-0.5), LPIPS Var(~0.0001-0.01)")
        score = (mse_variance * 1000) + (mse_mean * 100) + (lpips_mean * 50) + (lpips_variance * 500)
    else:
        score = (mse_variance * 1000) + (mse_mean * 100)
    
    return float(score)


def compute_flicker_index(frames: torch.Tensor) -> float:
    """
    Compute a flicker index based on second-order differences.
    
    Flicker = rapid back-and-forth changes
    We detect this by looking at: frame[t] - 2*frame[t+1] + frame[t+2]
    High values indicate oscillation (flicker).
    """
    if len(frames) < 3:
        return 0.0
    
    flicker_scores = []
    for t in range(len(frames) - 2):
        # Second derivative approximation
        second_diff = frames[t] - 2 * frames[t+1] + frames[t+2]
        flicker_scores.append(second_diff.abs().mean().item())
    
    return float(np.mean(flicker_scores))


# ============================================================
# Main Measurement Function
# ============================================================

def measure_video(
    frames: torch.Tensor,
    video_name: str = "video",
    compute_flow: bool = True,
    compute_lpips: bool = True
) -> VideoMetrics:
    """
    Compute all temporal consistency metrics for a video.
    
    Args:
        frames: [F, C, H, W] tensor in [0, 1]
        video_name: Name for identification
        compute_flow: Whether to compute optical flow metrics
        compute_lpips: Whether to compute LPIPS
    
    Returns:
        VideoMetrics dataclass with all measurements
    """
    F, C, H, W = frames.shape
    print(f"\nMeasuring video: {video_name}")
    print(f"Frames: {F}, Resolution: {H}x{W}")
    
    # Initialize metric computers
    lpips_metric = LPIPSMetric() if compute_lpips else None
    flow_estimator = OpticalFlowEstimator() if compute_flow else None
    
    frame_metrics = []
    all_flow_magnitudes = []
    
    # Process consecutive frame pairs
    for i in range(F - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        
        # Basic metrics
        mse = compute_mse(frame1, frame2)
        psnr = compute_psnr(frame1, frame2)
        
        # LPIPS
        lpips_val = None
        if lpips_metric is not None:
            lpips_val = lpips_metric.compute(frame1, frame2)
        
        # Optical flow
        flow_mag_mean = None
        flow_mag_std = None
        warp_error = None
        
        if flow_estimator is not None:
            flow = flow_estimator.compute_flow(frame1, frame2)
            if flow is not None:
                flow_stats = flow_estimator.compute_flow_stats(flow)
                flow_mag_mean = flow_stats['magnitude_mean']
                flow_mag_std = flow_stats['magnitude_std']
                all_flow_magnitudes.append(flow_stats['magnitude_mean'])
                
                # Warp error: warp frame1 by flow, compare to frame2
                warped = flow_estimator.warp_frame(frame1, flow)
                warp_error = compute_mse(warped, frame2)
        
        metrics = FramePairMetrics(
            frame_idx=i,
            mse=mse,
            psnr=psnr,
            lpips=lpips_val,
            flow_magnitude_mean=flow_mag_mean,
            flow_magnitude_std=flow_mag_std,
            warp_error=warp_error
        )
        frame_metrics.append(metrics)
        
        # Progress
        if (i + 1) % 5 == 0 or i == F - 2:
            print(f"  Processed {i + 1}/{F - 1} frame pairs")
    
    # Aggregate statistics
    mse_values = [m.mse for m in frame_metrics]
    psnr_values = [m.psnr for m in frame_metrics]
    lpips_values = [m.lpips for m in frame_metrics if m.lpips is not None]
    
    mean_lpips = float(np.mean(lpips_values)) if lpips_values else None
    std_lpips = float(np.std(lpips_values)) if lpips_values else None
    
    # Flow statistics
    mean_flow_mag = float(np.mean(all_flow_magnitudes)) if all_flow_magnitudes else None
    flow_mag_var = float(np.var(all_flow_magnitudes)) if all_flow_magnitudes else None
    
    # Temporal consistency score
    consistency_score = compute_temporal_consistency_score(frame_metrics)
    
    # Flicker index
    flicker_idx = compute_flicker_index(frames)
    
    video_metrics = VideoMetrics(
        video_name=video_name,
        num_frames=F,
        frame_metrics=frame_metrics,
        mean_mse=float(np.mean(mse_values)),
        std_mse=float(np.std(mse_values)),
        mean_psnr=float(np.mean(psnr_values)),
        mean_lpips=mean_lpips,
        std_lpips=std_lpips,
        temporal_consistency_score=consistency_score,
        mean_flow_magnitude=mean_flow_mag,
        flow_magnitude_variance=flow_mag_var,
    )
    
    return video_metrics


# ============================================================
# Reporting
# ============================================================

def print_metrics_summary(metrics: VideoMetrics):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print(f"METRICS SUMMARY: {metrics.video_name}")
    print("=" * 60)
    
    print(f"\nFrames: {metrics.num_frames}")
    
    print(f"\n--- Pixel Metrics ---")
    print(f"Mean MSE:  {metrics.mean_mse:.6f} (± {metrics.std_mse:.6f})")
    print(f"Mean PSNR: {metrics.mean_psnr:.2f} dB")
    
    if metrics.mean_lpips is not None:
        print(f"\n--- Perceptual Metrics ---")
        print(f"Mean LPIPS: {metrics.mean_lpips:.4f} (± {metrics.std_lpips:.4f})")
        print(f"  (lower = more similar between frames)")
    
    if metrics.mean_flow_magnitude is not None:
        print(f"\n--- Motion Metrics ---")
        print(f"Mean Flow Magnitude: {metrics.mean_flow_magnitude:.2f} px/frame")
        print(f"Flow Variance: {metrics.flow_magnitude_variance:.4f}")
        print(f"  (high variance = inconsistent motion)")
    
    print(f"\n--- Consistency Score ---")
    print(f"Temporal Consistency Score: {metrics.temporal_consistency_score:.4f}")
    print(f"  (lower = better, more temporally consistent)")
    
    # Interpretation
    print(f"\n--- Interpretation ---")
    if metrics.mean_mse < 0.001:
        print("• Very low inter-frame difference (possibly static)")
    elif metrics.mean_mse < 0.01:
        print("• Low inter-frame difference (smooth motion)")
    elif metrics.mean_mse < 0.05:
        print("• Moderate inter-frame difference")
    else:
        print("• High inter-frame difference (rapid change or flickering)")
    
    if metrics.std_mse / (metrics.mean_mse + 1e-8) > 0.5:
        print("• High MSE variance suggests inconsistent frame-to-frame changes")


def save_metrics(metrics: VideoMetrics, output_path: Path):
    """Save metrics to JSON."""
    # Convert to serializable dict
    data = {
        'video_name': metrics.video_name,
        'num_frames': metrics.num_frames,
        'mean_mse': metrics.mean_mse,
        'std_mse': metrics.std_mse,
        'mean_psnr': metrics.mean_psnr,
        'mean_lpips': metrics.mean_lpips,
        'std_lpips': metrics.std_lpips,
        'temporal_consistency_score': metrics.temporal_consistency_score,
        'mean_flow_magnitude': metrics.mean_flow_magnitude,
        'flow_magnitude_variance': metrics.flow_magnitude_variance,
        'frame_metrics': [asdict(fm) for fm in metrics.frame_metrics]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nMetrics saved to: {output_path}")


# ============================================================
# Batch Processing
# ============================================================

def measure_all_videos(base_dir: Path) -> List[VideoMetrics]:
    """
    Measure all video frame directories in a base directory.
    
    Expects structure like:
        base_dir/
            video1_frames/
                frame_000.png
                frame_001.png
                ...
            video2_frames/
                ...
    """
    base_dir = Path(base_dir)
    
    # Find all frame directories
    frame_dirs = [d for d in base_dir.iterdir() if d.is_dir() and 'frames' in d.name]
    
    if not frame_dirs:
        # Maybe frames are directly in base_dir
        frame_files = list(base_dir.glob('*.png')) + list(base_dir.glob('*.jpg'))
        if frame_files:
            frame_dirs = [base_dir]
    
    print(f"Found {len(frame_dirs)} video(s) to measure")
    
    all_metrics = []
    for frame_dir in sorted(frame_dirs):
        try:
            frames = load_frames(frame_dir)
            video_name = frame_dir.name.replace('_frames', '')
            metrics = measure_video(frames, video_name)
            all_metrics.append(metrics)
            print_metrics_summary(metrics)
            
            # Save individual metrics
            save_metrics(metrics, OUTPUT_DIR / f"{video_name}_metrics.json")
            
        except Exception as e:
            print(f"Error processing {frame_dir}: {e}")
    
    return all_metrics


def compare_videos(metrics_list: List[VideoMetrics]):
    """Compare metrics across multiple videos."""
    if len(metrics_list) < 2:
        return
    
    print("\n" + "=" * 60)
    print("VIDEO COMPARISON")
    print("=" * 60)
    
    # Sort by consistency score
    sorted_metrics = sorted(metrics_list, key=lambda m: m.temporal_consistency_score)
    
    print(f"\n{'Video':<30} {'MSE':<12} {'LPIPS':<12} {'Consistency':<12}")
    print("-" * 66)
    
    for m in sorted_metrics:
        lpips_str = f"{m.mean_lpips:.4f}" if m.mean_lpips else "N/A"
        print(f"{m.video_name:<30} {m.mean_mse:<12.6f} {lpips_str:<12} {m.temporal_consistency_score:<12.4f}")
    
    print(f"\nBest (most consistent): {sorted_metrics[0].video_name}")
    print(f"Worst (most flickery): {sorted_metrics[-1].video_name}")


# ============================================================
# Main
# ============================================================
'''
python experiments/05_measurement_pipeline.py --input outputs/01_baseline/corgi_beach_frames
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure temporal consistency of video frames")
    parser.add_argument("--input", type=str, default="outputs/01_baseline",
                        help="Directory containing frame folders or frames directly")
    parser.add_argument("--no-flow", action="store_true", help="Skip optical flow computation")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS computation")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    

    print(f"\nMeasuring videos in: {input_path}")
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        exit(1)
    
    # Check if it's a single video or multiple
    frame_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if frame_files and not subdirs:
        # Single video
        frames = load_frames(input_path)
        metrics = measure_video(
            frames, 
            input_path.name,
            compute_flow=not args.no_flow,
            compute_lpips=not args.no_lpips
        )
        print_metrics_summary(metrics)
        save_metrics(metrics, OUTPUT_DIR / f"{input_path.name}_metrics.json")
    else:
        # Multiple videos
        all_metrics = measure_all_videos(input_path)
        if len(all_metrics) > 1:
            compare_videos(all_metrics)
    
    print("\nMeasurement complete!")
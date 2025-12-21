"""
Experiment 05: Hyperparameter Ablation for Temporal Consistency

Systematically test different hyperparameter configurations and measure
their impact on temporal consistency.

Usage:
    python experiments/05_hyperparameter_ablation.py
    python experiments/05_hyperparameter_ablation.py --quick  # Fewer combinations
"""

import torch
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import sys

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = Path("outputs/05_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # Fixed seed for fair comparison

# ============================================================
# Test Prompts with Optimized Negative Prompts
# ============================================================

# Format: (name, prompt, negative_prompt, recommended_cfg, recommended_steps)
TEST_CASES = [
    {
        "name": "birds_flying",
        "prompt": "birds flying across a blue sky, nature documentary, smooth motion, consistent shapes",
        "negative": "flickering, morphing birds, changing shapes, unstable, jittery feathers, bad quality, blurry, distorted",
        "baseline_cfg": 7.5,
        "recommended_cfg": 5.0,
        "baseline_steps": 25,
        "recommended_steps": 35,
    },
    {
        "name": "corgi_beach",
        "prompt": "a corgi walking on the beach, sunset lighting, steady camera, smooth motion, high quality",
        "negative": "flickering water, unstable waves, jittery, morphing, handheld camera, shaky, bad quality, blurry, distorted",
        "baseline_cfg": 7.5,
        "recommended_cfg": 5.5,
        "baseline_steps": 25,
        "recommended_steps": 30,
    },
    {
        "name": "mig21_missile",
        "prompt": "MiG-21 fighter jet firing missile, smooth motion blur, cinematic, steady shot",
        "negative": "flickering, jittery, teleporting, inconsistent trail, morphing, bad quality, blurry, distorted",
        "baseline_cfg": 7.5,
        "recommended_cfg": 6.0,
        "baseline_steps": 25,
        "recommended_steps": 40,
    },
    {
        "name": "woman_waving",
        "prompt": "a woman waving her hand, portrait, studio lighting, smooth natural motion",
        "negative": "flickering hands, morphing fingers, jittery, distorted hands, bad quality, blurry, deformed",
        "baseline_cfg": 7.5,
        "recommended_cfg": 6.5,
        "baseline_steps": 25,
        "recommended_steps": 30,
    },
    {
        "name": "portrait",
        "prompt": "portrait of a man with glasses, professional photo, static pose, consistent lighting",
        "negative": "flickering, changing expression, morphing face, unstable features, bad quality, blurry, distorted",
        "baseline_cfg": 7.5,
        "recommended_cfg": 7.0,
        "baseline_steps": 25,
        "recommended_steps": 35,
    },
    {
        "name": "landscape",
        "prompt": "a beautiful mountain landscape, lake reflection, golden hour, still water, serene",
        "negative": "flickering water, rippling, moving clouds, windy, bad quality, blurry, distorted",
        "baseline_cfg": 7.5,
        "recommended_cfg": 7.0,
        "baseline_steps": 25,
        "recommended_steps": 25,
    },
]

# ============================================================
# Experiment Configurations
# ============================================================

def get_experiment_configs(quick_mode: bool = False) -> List[Dict]:
    """
    Generate experiment configurations.
    
    Each config tests a specific hypothesis about what improves consistency.
    """
    experiments = []
    
    for test_case in TEST_CASES:
        name = test_case["name"]
        
        # Experiment 1: Baseline (original settings)
        experiments.append({
            "experiment_id": f"{name}_baseline",
            "video_name": name,
            "prompt": test_case["prompt"].replace("smooth motion, ", "").replace("steady camera, ", ""),  # Original prompt
            "negative_prompt": "bad quality, blurry, distorted",  # Original negative
            "guidance_scale": test_case["baseline_cfg"],
            "num_inference_steps": test_case["baseline_steps"],
            "hypothesis": "baseline",
        })
        
        # Experiment 2: Recommended settings (full optimization)
        experiments.append({
            "experiment_id": f"{name}_recommended",
            "video_name": name,
            "prompt": test_case["prompt"],
            "negative_prompt": test_case["negative"],
            "guidance_scale": test_case["recommended_cfg"],
            "num_inference_steps": test_case["recommended_steps"],
            "hypothesis": "recommended",
        })
        
        if not quick_mode:
            # Experiment 3: Only lower CFG (isolate CFG impact)
            experiments.append({
                "experiment_id": f"{name}_low_cfg_only",
                "video_name": name,
                "prompt": test_case["prompt"].replace("smooth motion, ", "").replace("steady camera, ", ""),
                "negative_prompt": "bad quality, blurry, distorted",
                "guidance_scale": test_case["recommended_cfg"],
                "num_inference_steps": test_case["baseline_steps"],
                "hypothesis": "cfg_only",
            })
            
            # Experiment 4: Only more steps (isolate step count impact)
            experiments.append({
                "experiment_id": f"{name}_more_steps_only",
                "video_name": name,
                "prompt": test_case["prompt"].replace("smooth motion, ", "").replace("steady camera, ", ""),
                "negative_prompt": "bad quality, blurry, distorted",
                "guidance_scale": test_case["baseline_cfg"],
                "num_inference_steps": test_case["recommended_steps"],
                "hypothesis": "steps_only",
            })
            
            # Experiment 5: Only improved prompts (isolate prompt impact)
            experiments.append({
                "experiment_id": f"{name}_prompt_only",
                "video_name": name,
                "prompt": test_case["prompt"],
                "negative_prompt": test_case["negative"],
                "guidance_scale": test_case["baseline_cfg"],
                "num_inference_steps": test_case["baseline_steps"],
                "hypothesis": "prompt_only",
            })
    
    return experiments


# ============================================================
# Pipeline Management
# ============================================================

def load_pipeline():
    """Load AnimateDiff pipeline."""
    print("Loading motion adapter...")
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=DTYPE
    )
    
    print("Loading pipeline...")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=DTYPE
    )
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
        beta_schedule="linear"
    )
    
    pipe.to(DEVICE)
    pipe.enable_vae_slicing()
    
    return pipe


# ============================================================
# Generation
# ============================================================

def generate_video(pipe, config: Dict, output_dir: Path) -> Path:
    """Generate a single video with given configuration."""
    
    experiment_id = config["experiment_id"]
    print(f"\n{'='*60}")
    print(f"Generating: {experiment_id}")
    print(f"  CFG: {config['guidance_scale']}, Steps: {config['num_inference_steps']}")
    print(f"  Hypothesis: {config['hypothesis']}")
    print(f"{'='*60}")
    
    output = pipe(
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        num_frames=16,
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        width=512,
        height=512,
        generator=torch.Generator(DEVICE).manual_seed(SEED),
    )
    
    frames = output.frames[0]
    
    # Create experiment directory
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(exist_ok=True)
    
    # Save GIF
    gif_path = exp_dir / f"{experiment_id}.gif"
    export_to_gif(frames, str(gif_path))
    
    # Save frames
    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Saved to: {exp_dir}")
    
    return frames_dir


# ============================================================
# Measurement (inline to avoid import issues)
# ============================================================

def measure_video_quick(frames_dir: Path) -> Dict:
    """
    Quick measurement of temporal consistency.
    Returns key metrics only.
    """
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F
    
    # Load frames
    frame_files = sorted(frames_dir.glob("*.png"))
    frames = []
    for f in frame_files:
        img = Image.open(f).convert('RGB')
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(tensor)
    frames = torch.stack(frames)
    
    # Compute MSE between consecutive frames
    mse_values = []
    for i in range(len(frames) - 1):
        mse = F.mse_loss(frames[i], frames[i+1]).item()
        mse_values.append(mse)
    
    # Try LPIPS if available
    lpips_mean = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
        lpips_model.eval()
        
        lpips_values = []
        for i in range(len(frames) - 1):
            f1 = (frames[i].unsqueeze(0) * 2 - 1).to(DEVICE)
            f2 = (frames[i+1].unsqueeze(0) * 2 - 1).to(DEVICE)
            with torch.no_grad():
                lpips_val = lpips_model(f1, f2).item()
            lpips_values.append(lpips_val)
        lpips_mean = float(np.mean(lpips_values))
        lpips_std = float(np.std(lpips_values))
        
        del lpips_model
        torch.cuda.empty_cache()
    except ImportError:
        lpips_std = None
    
    # Compute flow if OpenCV available
    flow_variance = None
    try:
        import cv2
        flow_mags = []
        for i in range(len(frames) - 1):
            gray1 = (frames[i].mean(dim=0) * 255).numpy().astype(np.uint8)
            gray2 = (frames[i+1].mean(dim=0) * 255).numpy().astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            flow_mags.append(mag)
        flow_variance = float(np.var(flow_mags))
        flow_mean = float(np.mean(flow_mags))
    except ImportError:
        flow_mean = None
    
    return {
        "mse_mean": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "flow_mean": flow_mean,
        "flow_variance": flow_variance,
    }


# ============================================================
# Results Analysis
# ============================================================

def analyze_results(results: List[Dict]) -> str:
    """Generate analysis report from results."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("HYPERPARAMETER ABLATION RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by video
    videos = {}
    for r in results:
        video_name = r["config"]["video_name"]
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(r)
    
    # Overall comparison table
    lines.append("SUMMARY TABLE")
    lines.append("-" * 80)
    header = f"{'Experiment':<35} {'MSE':<10} {'LPIPS':<10} {'Flow Var':<10} {'Better?':<10}"
    lines.append(header)
    lines.append("-" * 80)
    
    improvements = {"cfg_only": 0, "steps_only": 0, "prompt_only": 0, "recommended": 0}
    total_per_hypothesis = {"cfg_only": 0, "steps_only": 0, "prompt_only": 0, "recommended": 0}
    
    for video_name, video_results in sorted(videos.items()):
        # Find baseline
        baseline = next((r for r in video_results if r["config"]["hypothesis"] == "baseline"), None)
        
        for r in sorted(video_results, key=lambda x: x["config"]["hypothesis"]):
            exp_id = r["config"]["experiment_id"]
            metrics = r["metrics"]
            hypothesis = r["config"]["hypothesis"]
            
            mse_str = f"{metrics['mse_mean']:.4f}" if metrics['mse_mean'] else "N/A"
            lpips_str = f"{metrics['lpips_mean']:.3f}" if metrics['lpips_mean'] else "N/A"
            flow_str = f"{metrics['flow_variance']:.3f}" if metrics['flow_variance'] else "N/A"
            
            # Determine if better than baseline
            better = ""
            if baseline and hypothesis != "baseline":
                total_per_hypothesis[hypothesis] = total_per_hypothesis.get(hypothesis, 0) + 1
                
                baseline_score = baseline["metrics"]["mse_mean"]
                current_score = metrics["mse_mean"]
                
                if current_score < baseline_score * 0.95:  # 5% improvement threshold
                    better = "✓"
                    improvements[hypothesis] = improvements.get(hypothesis, 0) + 1
                elif current_score > baseline_score * 1.05:
                    better = "✗"
                else:
                    better = "~"
            
            lines.append(f"{exp_id:<35} {mse_str:<10} {lpips_str:<10} {flow_str:<10} {better:<10}")
        
        lines.append("")
    
    # Summary statistics
    lines.append("")
    lines.append("IMPROVEMENT RATES BY HYPOTHESIS")
    lines.append("-" * 40)
    for hyp in ["cfg_only", "steps_only", "prompt_only", "recommended"]:
        if total_per_hypothesis.get(hyp, 0) > 0:
            rate = improvements.get(hyp, 0) / total_per_hypothesis[hyp] * 100
            lines.append(f"{hyp:<20}: {improvements.get(hyp, 0)}/{total_per_hypothesis[hyp]} ({rate:.0f}%)")
    
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter ablation study")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick mode: only baseline vs recommended")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific videos to test (default: all)")
    args = parser.parse_args()
    
    # Get experiment configurations
    configs = get_experiment_configs(quick_mode=args.quick)
    
    # Filter by video if specified
    if args.videos:
        configs = [c for c in configs if c["video_name"] in args.videos]
    
    print(f"Running {len(configs)} experiments...")
    
    # Load pipeline
    pipe = load_pipeline()
    
    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]", end="")
        
        # Generate
        frames_dir = generate_video(pipe, config, OUTPUT_DIR)
        
        # Measure
        print("  Measuring...")
        metrics = measure_video_quick(frames_dir)
        
        results.append({
            "config": config,
            "metrics": metrics,
        })
        
        # Save intermediate results
        results_path = OUTPUT_DIR / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Analyze and save report
    report = analyze_results(results)
    print("\n" + report)
    
    report_path = OUTPUT_DIR / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
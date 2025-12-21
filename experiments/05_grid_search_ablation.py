"""
Experiment 05: Systematic Grid Search Ablation

Proper experimental design: vary one parameter at a time, measure effects.

Phase 1: CFG ablation (steps=25, baseline prompts)
Phase 2: Steps ablation (CFG=7.5, baseline prompts)
Phase 3: Prompt ablation (CFG=7.5, steps=25, baseline vs enhanced prompts)

Usage:
    python experiments/05_grid_search_ablation.py
    python experiments/05_grid_search_ablation.py --phase cfg
    python experiments/05_grid_search_ablation.py --phase steps
    python experiments/05_grid_search_ablation.py --phase prompt
    python experiments/05_grid_search_ablation.py --video portrait --phase cfg
"""

import torch
import argparse
import json
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = Path("outputs/05_grid_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid search parameters
CFG_VALUES = [5.0, 6.0, 7.0, 7.5, 8.0, 9.0]
STEPS_VALUES = [15, 20, 25, 30, 40, 50]

# Defaults
DEFAULT_CFG = 7.5
DEFAULT_STEPS = 25

# Fixed parameters
NUM_FRAMES = 16
HEIGHT = 512
WIDTH = 512
SEED = 42

# ============================================================
# Test Videos Configuration
# ============================================================

TEST_VIDEOS = {
    "birds_flying": {
        "prompt_baseline": "birds flying across a blue sky, nature documentary",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "birds flying across a blue sky, nature documentary, smooth motion, consistent shapes",
        "negative_enhanced": "flickering, morphing birds, changing shapes, unstable, jittery feathers, bad quality, blurry, distorted",
    },
    "corgi_beach": {
        "prompt_baseline": "a corgi walking on the beach, sunset lighting, high quality",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "a corgi walking on the beach, sunset lighting, steady camera, smooth motion, high quality",
        "negative_enhanced": "flickering water, unstable waves, jittery, morphing, shaky, bad quality, blurry, distorted",
    },
    "mig21_missile": {
        "prompt_baseline": "MiG-21 fighter jet firing missile, action shot, cinematic",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "MiG-21 fighter jet firing missile, smooth motion blur, cinematic, steady tracking shot",
        "negative_enhanced": "flickering, jittery, teleporting, inconsistent trail, morphing, bad quality, blurry, distorted",
    },
    "woman_waving": {
        "prompt_baseline": "a woman waving her hand, portrait, studio lighting",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "a woman waving her hand, portrait, studio lighting, smooth natural motion",
        "negative_enhanced": "flickering hands, morphing fingers, jittery, distorted hands, bad quality, blurry, deformed",
    },
    "portrait": {
        "prompt_baseline": "portrait of a man with glasses, professional photo, static pose",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "portrait of a man with glasses, professional photo, static pose, consistent lighting",
        "negative_enhanced": "flickering, changing expression, morphing face, unstable features, bad quality, blurry, distorted",
    },
    "landscape": {
        "prompt_baseline": "a beautiful mountain landscape, lake reflection, golden hour, serene",
        "negative_baseline": "bad quality, blurry, distorted",
        "prompt_enhanced": "a beautiful mountain landscape, lake reflection, golden hour, still water, serene",
        "negative_enhanced": "flickering water, rippling, moving clouds, windy, bad quality, blurry, distorted",
    },
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    video_name: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    phase: str
    seed: int = SEED
    num_frames: int = NUM_FRAMES
    height: int = HEIGHT
    width: int = WIDTH


# ============================================================
# Pipeline Loading
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
        torch_dtype=DTYPE,
    )
    
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        steps_offset=1,
        clip_sample=False,
    )
    
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
    print("Pipeline loaded successfully")
    return pipe


# ============================================================
# Generation
# ============================================================

def generate_video(pipe, config: ExperimentConfig) -> List:
    """Generate video frames for given configuration."""
    generator = torch.manual_seed(config.seed)
    
    output = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        num_frames=config.num_frames,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        height=config.height,
        width=config.width,
        generator=generator,
    )
    
    return output.frames[0]


def save_experiment(frames: List, config: ExperimentConfig, output_dir: Path) -> Path:
    """Save frames, GIF, and config."""
    exp_dir = output_dir / config.experiment_id
    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:04d}.png")
    
    gif_path = exp_dir / f"{config.experiment_id}.gif"
    export_to_gif(frames, gif_path)
    
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    return exp_dir


# ============================================================
# Ablation Runners
# ============================================================

def run_cfg_ablation(pipe, video_name: str, video_config: dict, output_dir: Path) -> List[ExperimentConfig]:
    """Phase 1: Vary CFG, hold steps constant, use baseline prompts."""
    configs = []
    
    for cfg in CFG_VALUES:
        experiment_id = f"{video_name}_cfg{cfg:.1f}_steps{DEFAULT_STEPS}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            video_name=video_name,
            prompt=video_config["prompt_baseline"],
            negative_prompt=video_config["negative_baseline"],
            guidance_scale=cfg,
            num_inference_steps=DEFAULT_STEPS,
            phase="cfg_ablation",
        )
        
        exp_dir = output_dir / experiment_id
        if (exp_dir / "config.json").exists():
            print(f"  Skipping {experiment_id} (already exists)")
            configs.append(config)
            continue
        
        print(f"  Generating: CFG={cfg}, Steps={DEFAULT_STEPS}")
        frames = generate_video(pipe, config)
        save_experiment(frames, config, output_dir)
        configs.append(config)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return configs


def run_steps_ablation(pipe, video_name: str, video_config: dict, output_dir: Path) -> List[ExperimentConfig]:
    """Phase 2: Vary steps, hold CFG constant, use baseline prompts."""
    configs = []
    
    for steps in STEPS_VALUES:
        experiment_id = f"{video_name}_cfg{DEFAULT_CFG:.1f}_steps{steps}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            video_name=video_name,
            prompt=video_config["prompt_baseline"],
            negative_prompt=video_config["negative_baseline"],
            guidance_scale=DEFAULT_CFG,
            num_inference_steps=steps,
            phase="steps_ablation",
        )
        
        exp_dir = output_dir / experiment_id
        if (exp_dir / "config.json").exists():
            print(f"  Skipping {experiment_id} (already exists)")
            configs.append(config)
            continue
        
        print(f"  Generating: CFG={DEFAULT_CFG}, Steps={steps}")
        frames = generate_video(pipe, config)
        save_experiment(frames, config, output_dir)
        configs.append(config)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return configs


def run_prompt_ablation(pipe, video_name: str, video_config: dict, output_dir: Path) -> List[ExperimentConfig]:
    """Phase 3: Compare baseline vs enhanced prompts at default CFG/steps."""
    configs = []
    
    prompt_variants = [
        ("baseline", video_config["prompt_baseline"], video_config["negative_baseline"]),
        ("enhanced", video_config["prompt_enhanced"], video_config["negative_enhanced"]),
    ]
    
    for variant_name, prompt, negative in prompt_variants:
        experiment_id = f"{video_name}_cfg{DEFAULT_CFG:.1f}_steps{DEFAULT_STEPS}_prompt_{variant_name}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            video_name=video_name,
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=DEFAULT_CFG,
            num_inference_steps=DEFAULT_STEPS,
            phase="prompt_ablation",
        )
        
        exp_dir = output_dir / experiment_id
        if (exp_dir / "config.json").exists():
            print(f"  Skipping {experiment_id} (already exists)")
            configs.append(config)
            continue
        
        print(f"  Generating: Prompt={variant_name}")
        frames = generate_video(pipe, config)
        save_experiment(frames, config, output_dir)
        configs.append(config)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return configs


def run_grid_search(
    pipe,
    phase: str = "all",
    video_filter: Optional[str] = None,
    output_dir: Path = OUTPUT_DIR
) -> List[ExperimentConfig]:
    """Run the grid search ablation study."""
    
    all_configs = []
    
    videos = TEST_VIDEOS
    if video_filter:
        videos = {k: v for k, v in TEST_VIDEOS.items() if video_filter in k}
    
    for video_name, video_config in videos.items():
        print(f"\n{'='*60}")
        print(f"Video: {video_name}")
        print(f"{'='*60}")
        
        if phase in ["all", "cfg"]:
            print("\n[Phase 1] CFG Ablation (baseline prompts)")
            configs = run_cfg_ablation(pipe, video_name, video_config, output_dir)
            all_configs.extend(configs)
        
        if phase in ["all", "steps"]:
            print("\n[Phase 2] Steps Ablation (baseline prompts)")
            configs = run_steps_ablation(pipe, video_name, video_config, output_dir)
            all_configs.extend(configs)
        
        if phase in ["all", "prompt"]:
            print("\n[Phase 3] Prompt Ablation (baseline vs enhanced)")
            configs = run_prompt_ablation(pipe, video_name, video_config, output_dir)
            all_configs.extend(configs)
    
    return all_configs


# ============================================================
# Manifest Generation
# ============================================================

def generate_manifest(output_dir: Path):
    """Generate manifest of all experiments."""
    manifest = {
        "grid_params": {
            "cfg_values": CFG_VALUES,
            "steps_values": STEPS_VALUES,
            "default_cfg": DEFAULT_CFG,
            "default_steps": DEFAULT_STEPS,
        },
        "experiments": [],
    }
    
    for exp_dir in sorted(output_dir.iterdir()):
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            manifest["experiments"].append({
                "experiment_id": config["experiment_id"],
                "video_name": config["video_name"],
                "cfg": config["guidance_scale"],
                "steps": config["num_inference_steps"],
                "phase": config["phase"],
            })
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total experiments: {len(manifest['experiments'])}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Systematic Grid Search Ablation")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "cfg", "steps", "prompt"],
                        help="Which ablation phase to run")
    parser.add_argument("--video", type=str, default=None,
                        help="Run only for specific video (partial match)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SYSTEMATIC GRID SEARCH ABLATION")
    print("=" * 60)
    print(f"\nPhase: {args.phase}")
    print(f"Video filter: {args.video or 'all'}")
    print(f"Output: {output_dir}")
    print(f"\nCFG values: {CFG_VALUES}")
    print(f"Steps values: {STEPS_VALUES}")
    
    input("\nPress Enter to start (or Ctrl+C to cancel)...")
    
    pipe = load_pipeline()
    
    start_time = datetime.now()
    configs = run_grid_search(pipe, args.phase, args.video, output_dir)
    end_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    
    generate_manifest(output_dir)
    
    print(f"\nTotal time: {end_time - start_time}")
    print(f"Experiments generated: {len(configs)}")
    print(f"\nNext step: python experiments/06_measure_grid_search.py")


if __name__ == "__main__":
    main()

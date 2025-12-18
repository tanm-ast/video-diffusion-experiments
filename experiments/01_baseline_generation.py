"""
Baseline video generation with AnimateDiff.
Goal: Get a working baseline, understand the pipeline, generate test videos
for later analysis.

Hardware: RTX 3060 12GB VRAM
"""

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
import os
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("outputs/01_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
DEVICE = "cuda"
DTYPE = torch.float16  # fp16 fits comfortably in 12GB

# Generation parameters
DEFAULT_CONFIG = {
    "num_frames": 16,           # Start small with 16 frames
    "num_inference_steps": 25,  # Moderate quality at 25 inference steps
    "guidance_scale": 7.5,      # CFG scale - we'll experiment with this, High classifier-free guidance strengthens per-frame prompt adherence, which can fight temporal smoothing.
    "width": 512,
    "height": 512,
}


# Test prompts - variety to observe different failure modes
TEST_PROMPTS = [
    # Motion-heavy (likely to show temporal inconsistency)
    ("corgi_beach", "a corgi walking on the beach, sunset lighting, high quality"),
    ("woman_waving", "a pretty woman waving her hand, portrait, studio lighting"),
    
    # Static scene (should be more consistent)
    ("landscape", "a beautiful mountain landscape, lake reflection, golden hour"),
    ("portrait", "portrait of a man with glasses, professional photo"),
    
    # Complex motion (stress test)
    ("birds_flying", "birds flying across a blue sky, nature documentary"),
    ("mig21_missile", "A Mig-21 firing a missile, cloudy pink sky in the background, high quality"),
]

NEGATIVE_PROMPT = "bad quality, blurry, distorted, ugly, deformed"


# ============================================================
# Pipeline Setup
# ============================================================

def load_pipeline():
    """Load AnimateDiff pipeline with memory optimizations."""
    print("Loading motion adapter...")
    # Motion adapter contains the temporal attention layers
    # Architectural choice that makes it video instead of independent images
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=DTYPE
    )

    print("Loading pipeline...")
    # Base model is a standard SD 1.5 checkpoint
    # Generates only images.
    # AnimateDiff wraps it and injects temporal attention
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=DTYPE
    )

    # Better scheduler for quality (vs default DDPM/DDIM)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
        beta_schedule="linear"
    )

    pipe.to(DEVICE)

    # Enable memory optimizations
    pipe.enable_vae_slicing()

    return pipe

def generate_video(pipe, prompt, name, seed=42, **kwargs):
    """Generate a single video and save outputs."""
    config = {**DEFAULT_CONFIG, **kwargs}
    
    print(f"\nGenerating: {name}")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"  Config: {config}")

    # Generate
    # num_frames: how many frames (more = longer video, more VRAM)
    # num_inference_steps: denoising steps (more = higher quality, slower)
    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        generator=torch.Generator(DEVICE).manual_seed(seed),
        **config
    )

    # Save
    frames = output.frames[0]

    # Save as GIF
    gif_path = OUTPUT_DIR / f"{name}.gif"
    export_to_gif(frames, str(gif_path))
    print(f"Saved GIF: {gif_path}")


    # Save individual frames for analysis
    frame_dir = OUTPUT_DIR / f"{name}_frames"
    frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frame_dir / f"frame_{i:03d}.png")
    print(f"Saved {len(frames)} individual frames to {frame_dir}")

    return frames


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pipe = load_pipeline()
    
    print("\n" + "=" * 60)
    print("Generating baseline videos")
    print("=" * 60)
    
    for name, prompt in TEST_PROMPTS:
        generate_video(pipe, prompt, name)
    
    print("\n" + "=" * 60)
    print("Baseline generation complete!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Memory cleanup
    del pipe
    torch.cuda.empty_cache()

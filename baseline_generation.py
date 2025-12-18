"""
Baseline video generation with AnimateDiff.
Goal: Get something running, see the outputs, understand the pipeline.
"""

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
import os

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Device setup
device = "cuda"
dtype = torch.float16  # fp16 fits comfortably in 12GB

print("Loading motion adapter...")
# Motion adapter contains the temporal attention layers
# This is THE thing that makes it video instead of independent images
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=dtype
)

print("Loading pipeline...")
# Base model is a standard SD 1.5 checkpoint
# AnimateDiff wraps it and injects temporal attention
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=dtype
)

# Better scheduler for quality
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="linspace",
    beta_schedule="linear"
)

pipe.to(device)

# Enable memory optimizations
pipe.enable_vae_slicing()

print("Generating video...")

# Simple test prompt
prompt = "A Mig-21 firing a missile, cloudy pink sky in the background, high quality"
negative_prompt = "bad quality, blurry, distorted"

# Generate
# num_frames: how many frames (more = longer video, more VRAM)
# num_inference_steps: denoising steps (more = higher quality, slower)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=16,           # Start small
    num_inference_steps=25,  # Moderate quality
    guidance_scale=7.5,      # CFG scale - we'll experiment with this
    width=512,
    height=512,
    generator=torch.Generator(device).manual_seed(42)  # Reproducibility
)

# Save
frames = output.frames[0]
export_to_gif(frames, "outputs/baseline_001.gif")
print(f"Saved to outputs/baseline_001.gif")

# Also save individual frames for analysis
for i, frame in enumerate(frames):
    frame.save(f"outputs/baseline_001_frame_{i:03d}.png")
print(f"Saved {len(frames)} individual frames")
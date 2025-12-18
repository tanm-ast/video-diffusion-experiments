"""
Inspect the AnimateDiff architecture to understand where temporal
attention lives and how it interacts with the spatial layers.

This is the "read the blueprint" step before you start modifying.
"""

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter

dtype = torch.float32 # Load in float32 for inspection (we won't run inference)


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

unet = pipe.unet

print("=" * 60)
print("UNet Architecture Overview")
print("=" * 60)

# Count parameters
total_params = sum(p.numel() for p in unet.parameters())
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"Total parameters: {total_params / 1e6:.1f}M")
print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

print("\n" + "=" * 60)
print("Looking for Temporal Attention Layers")
print("=" * 60)

# Find all attention-related modules
temporal_modules = []
spatial_modules = []

for name, module in unet.named_modules():
    module_type = type(module).__name__
    
    # Temporal attention modules have specific naming in AnimateDiff
    if "temporal" in name.lower() or "motion" in name.lower():
        temporal_modules.append((name, module_type))
    elif "attn" in name.lower() and "temporal" not in name.lower():
        # Only track the main attention classes, not submodules
        if module_type in ["Attention", "CrossAttention", "SelfAttention"]:
            spatial_modules.append((name, module_type))

print(f"\nFound {len(temporal_modules)} temporal-related modules:")
for name, mtype in temporal_modules[:20]:  # First 20
    print(f"  {mtype}: {name}")
if len(temporal_modules) > 20:
    print(f"  ... and {len(temporal_modules) - 20} more")

print(f"\nFound {len(spatial_modules)} spatial attention modules:")
for name, mtype in spatial_modules[:10]:  # First 10
    print(f"  {mtype}: {name}")

print("\n" + "=" * 60)
print("Key Insight")
print("=" * 60)
print("""
The temporal attention layers are INSERTED between the existing
spatial attention layers. During inference:

1. Spatial self-attention: Each frame attends to itself
   (handles per-frame coherence)

2. Temporal attention: Same spatial position attends across frames
   (handles cross-frame consistency)

3. Cross-attention: Text conditioning
   (handles prompt adherence)

Temporal inconsistency typically means the temporal attention
is too weak relative to the spatial signal, OR the temporal
attention is learning spurious patterns.
""")

# Let's look at one temporal block in detail
print("\n" + "=" * 60)
print("Detailed look at a temporal attention block")
print("=" * 60)

for name, module in unet.named_modules():
    if "motion_modules" in name and "temporal_transformer" in name:
        if hasattr(module, 'proj_in') or 'attn' in name:
            print(f"\n{name}:")
            print(f"  Type: {type(module).__name__}")
            if hasattr(module, 'heads'):
                print(f"  Attention heads: {module.heads}")
            if hasattr(module, 'to_q'):
                print(f"  Query dim: {module.to_q.in_features} -> {module.to_q.out_features}")
        break
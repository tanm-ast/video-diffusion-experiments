"""
Experiment 03: Trace AnimateDiff Forward Pass

Use forward hooks to understand exactly what happens during inference.
This reveals:
- Execution order of modules
- Tensor shapes at each stage  
- How temporal attention reshapes tensors

Run this AFTER 01_baseline_generation.py (uses cached models).
"""

import torch
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from utils.forward_tracer import ForwardTracer

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = Path("outputs/03_traces")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load Model
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
    return pipe

# ============================================================
# Tracing Experiments
# ============================================================

def trace_unet_forward(pipe):
    """
    Trace a single UNet forward pass.
    
    The UNet takes:
    - sample: noisy latent [B, C, F, H, W]
    - timestep: current noise level
    - encoder_hidden_states: text embeddings [B, seq_len, dim]
    """
    print("\n" + "=" * 60)
    print("TRACING UNET FORWARD PASS")
    print("=" * 60)
    
    unet = pipe.unet
    
    # Create dummy inputs matching actual inference
    batch_size = 1
    num_frames = 16
    height, width = 512, 512
    latent_h, latent_w = height // 8, width // 8  # VAE downsamples 8x
    
    # Latent shape: [B, 4, F, H/8, W/8]
    sample = torch.randn(
        batch_size, 4, num_frames, latent_h, latent_w,
        device=DEVICE, dtype=DTYPE
    )
    
    # Timestep (scalar or batch)
    timestep = torch.tensor([500], device=DEVICE, dtype=DTYPE)
    
    # Text embeddings: [B, 77, 768] for SD 1.5
    encoder_hidden_states = torch.randn(
        batch_size, 77, 768,
        device=DEVICE, dtype=DTYPE
    )
    
    print(f"Input sample shape: {sample.shape}")
    print(f"Timestep: {timestep}")
    print(f"Text embedding shape: {encoder_hidden_states.shape}")
    
    # Create tracer
    tracer = ForwardTracer(unet, trace_depth=5)  # Limit depth to avoid clutter
    
    # Run traced forward
    with torch.no_grad():
        output = tracer.trace(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
    
    print(f"Output shape: {output.sample.shape}")
    
    # Save full report
    tracer.save_report(str(OUTPUT_DIR / "unet_trace.txt"))
    print(f"Full trace saved to: {OUTPUT_DIR / 'unet_trace.txt'}")
    
    return tracer


def analyze_attention_modules(tracer):
    """Analyze attention module shapes."""
    print("\n" + "=" * 60)
    print("ATTENTION MODULE ANALYSIS")
    print("=" * 60)
    
    # Find all attention modules
    spatial_attn = []
    temporal_attn = []
    
    for name, trace in tracer.traces.items():
        if trace.class_name == "Attention":
            if "motion_modules" in name:
                temporal_attn.append((name, trace))
            elif "attentions" in name:
                spatial_attn.append((name, trace))
    
    print(f"\nSpatial attention modules: {len(spatial_attn)}")
    print(f"Temporal attention modules: {len(temporal_attn)}")
    
    # Examine first spatial attention
    if spatial_attn:
        name, trace = spatial_attn[0]
        print(f"\n--- First Spatial Attention ---")
        print(f"Name: {name}")
        print(f"Input shape:  {trace.input_shapes}")
        print(f"Output shape: {trace.output_shapes}")
        
    # Examine first temporal attention
    if temporal_attn:
        name, trace = temporal_attn[0]
        print(f"\n--- First Temporal Attention ---")
        print(f"Name: {name}")
        print(f"Input shape:  {trace.input_shapes}")
        print(f"Output shape: {trace.output_shapes}")
        
        # Interpret the shape
        if trace.input_shapes:
            shape = trace.input_shapes[0]
            if len(shape) == 3:
                B_HW, F, C = shape
                print(f"\nInterpretation:")
                print(f"  Batch*Height*Width: {B_HW}")
                print(f"  Num Frames: {F}")
                print(f"  Channels: {C}")
                print(f"  -> Each spatial position attends across {F} frames")


def trace_motion_module_detail(pipe):
    """
    Detailed trace of a single motion module to understand
    the reshape operations.
    """
    print("\n" + "=" * 60)
    print("MOTION MODULE DETAILED TRACE")
    print("=" * 60)
    
    # Get first motion module
    motion_module = pipe.unet.down_blocks[0].motion_modules[0]
    
    print(f"Module type: {type(motion_module).__name__}")
    print(f"\nSubmodules:")
    for name, mod in motion_module.named_children():
        print(f"  {name}: {type(mod).__name__}")
    
    # Trace just this module
    tracer = ForwardTracer(motion_module)
    
    # Input to motion module: [B, C, F, H, W] but flattened
    # Actually it receives [B*F, C, H, W] from spatial processing
    # then internally handles frame dimension
    
    # Check the actual expected input by looking at the source
    # For AnimateDiffTransformer3D, input is [B, C, F, H, W]
    sample = torch.randn(1, 320, 16, 64, 64, device=DEVICE, dtype=DTYPE)
    
    print(f"\nInput shape: {sample.shape}")
    
    try:
        with torch.no_grad():
            # Need to provide num_frames for some versions
            output = tracer.trace(sample, num_frames=16)
        print(f"Output shape: {output.shape}")
        tracer.print_summary()
    except Exception as e:
        print(f"Direct trace failed: {e}")
        print("This is expected - motion modules may need specific calling convention")
    
    tracer.remove_hooks()


def compare_execution_order(tracer):
    """
    Show how spatial and temporal processing interleave.
    """
    print("\n" + "=" * 60)
    print("SPATIAL vs TEMPORAL EXECUTION ORDER")
    print("=" * 60)
    
    print("\nFirst 50 modules in execution order:\n")
    
    for i, name in enumerate(tracer.execution_order[:50]):
        trace = tracer.traces[name]
        
        # Color coding (conceptual)
        if "motion_modules" in name:
            prefix = "[TEMPORAL]"
        elif "attentions" in name:
            prefix = "[SPATIAL] "
        elif "resnets" in name:
            prefix = "[RESNET]  "
        else:
            prefix = "          "
        
        # Simplify name for display
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
        
        print(f"{i:3d}. {prefix} {short_name:<30} {trace.class_name}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pipe = load_pipeline()
    
    # Full UNet trace
    tracer = trace_unet_forward(pipe)
    
    # Analyze attention
    analyze_attention_modules(tracer)
    
    # Show execution order
    compare_execution_order(tracer)
    
    # Clean up
    tracer.remove_hooks()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Open outputs/03_traces/unet_trace.txt for full trace
2. Look for shape changes at motion_module boundaries
3. Verify temporal attention input shape is [B*H*W, F, C]
4. This confirms each spatial position attends across frames

Key insight: The tensor reshape from [B,C,F,H,W] to [B*H*W,F,C]
is what makes temporal attention "per-position across frames"
rather than "global spatiotemporal attention".
""")
    
    del pipe
    torch.cuda.empty_cache()

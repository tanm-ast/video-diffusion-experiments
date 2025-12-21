"""
Experiment 04: Trace AnimateDiff Forward Pass

Use forward hooks to understand exactly what happens during inference.
This reveals:
- Execution order of modules
- Tensor shapes at each stage  
- How temporal attention reshapes tensors

Strategy: Hook into an actual pipeline generation rather than calling
the UNet directly with synthetic inputs. This ensures we see the real
tensor shapes that the pipeline produces.
"""

import torch
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"
DTYPE = torch.float16
OUTPUT_DIR = Path("outputs/03_traces")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Lightweight Tracer (inline to avoid import issues)
# ============================================================

@dataclass
class TraceRecord:
    name: str
    class_name: str
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    execution_order: int


class LightweightTracer:
    """
    Simple tracer that records shapes during forward pass.
    Designed to be robust to complex module interactions.
    """
    
    def __init__(self, model, module_filter=None):
        self.model = model
        self.module_filter = module_filter
        self.traces: OrderedDict[str, TraceRecord] = OrderedDict()
        self.execution_order: List[str] = []
        self.hooks = []
        self._counter = 0
        
    def _get_shapes(self, x):
        if x is None:
            return [()]
        if isinstance(x, torch.Tensor):
            return [tuple(x.shape)]
        if isinstance(x, (tuple, list)):
            shapes = []
            for t in x:
                if isinstance(t, torch.Tensor):
                    shapes.append(tuple(t.shape))
                elif t is None:
                    shapes.append(())
            return shapes if shapes else [()]
        return [()]
    
    def _make_hook(self, name):
        def hook(module, inp, out):
            record = TraceRecord(
                name=name,
                class_name=module.__class__.__name__,
                input_shapes=self._get_shapes(inp),
                output_shapes=self._get_shapes(out),
                execution_order=self._counter
            )
            self.traces[name] = record
            self.execution_order.append(name)
            self._counter += 1
        return hook
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self.module_filter and not self.module_filter(name, module):
                continue
            h = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)
    
    def clear(self):
        self.traces.clear()
        self.execution_order.clear()
        self._counter = 0
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def print_summary(self, max_items=100):
        print(f"\n{'Ord':<5} {'Module Name':<55} {'Type':<25} {'Input':<25} {'Output':<25}")
        print("-" * 135)
        
        for i, name in enumerate(self.execution_order[:max_items]):
            trace = self.traces[name]
            short_name = name if len(name) <= 55 else "..." + name[-52:]
            in_shape = str(trace.input_shapes[0]) if trace.input_shapes else "?"
            out_shape = str(trace.output_shapes[0]) if trace.output_shapes else "?"
            
            # Truncate shapes for display
            in_shape = in_shape[:23] + ".." if len(in_shape) > 25 else in_shape
            out_shape = out_shape[:23] + ".." if len(out_shape) > 25 else out_shape
            
            print(f"{trace.execution_order:<5} {short_name:<55} {trace.class_name:<25} {in_shape:<25} {out_shape:<25}")
        
        if len(self.execution_order) > max_items:
            print(f"... and {len(self.execution_order) - max_items} more modules")


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
    pipe.enable_vae_slicing()
    
    return pipe


# ============================================================
# Tracing Experiments  
# ============================================================

def trace_during_generation(pipe):
    """
    Trace the UNet during an actual generation.
    This captures real tensor shapes as the pipeline uses them.
    """
    print("\n" + "=" * 60)
    print("TRACING UNET DURING ACTUAL GENERATION")
    print("=" * 60)
    
    # Create tracer for UNet only
    # Filter to key module types to reduce noise
    def important_modules(name, module):
        class_name = module.__class__.__name__
        # Focus on attention, motion, and key structural modules
        return class_name in [
            "Attention", 
            "AnimateDiffTransformer3D",
            "BasicTransformerBlock",
            "TransformerTemporalModel",
            "Transformer2DModel",
            "CrossAttnDownBlock2D",
            "CrossAttnUpBlock2D",
            "UNetMidBlock2DCrossAttn",
            "ResnetBlock2D",
        ]
    
    tracer = LightweightTracer(pipe.unet, module_filter=important_modules)
    tracer.register_hooks()
    
    print("Running generation with tracing enabled...")
    print("(This will be slow - tracing every module)\n")
    
    # Run a short generation (fewer steps for speed)
    with torch.no_grad():
        output = pipe(
            prompt="a cat walking",
            num_frames=16,
            num_inference_steps=5,  # Minimal steps for tracing
            guidance_scale=7.5,
            width=256,  # Smaller for faster tracing
            height=256,
            generator=torch.Generator(DEVICE).manual_seed(42),
            output_type="pt"  # Return tensors
        )
    
    print(f"Generation complete. Traced {len(tracer.traces)} modules.\n")
    
    return tracer


def analyze_attention_shapes(tracer):
    """Analyze attention module shapes to understand temporal vs spatial."""
    print("\n" + "=" * 60)
    print("ATTENTION SHAPE ANALYSIS")
    print("=" * 60)
    
    spatial_attn = []
    temporal_attn = []
    
    for name, trace in tracer.traces.items():
        if trace.class_name == "Attention":
            if "motion_modules" in name:
                temporal_attn.append((name, trace))
            elif "attentions" in name:
                spatial_attn.append((name, trace))
    
    print(f"\nFound {len(spatial_attn)} spatial attention modules")
    print(f"Found {len(temporal_attn)} temporal attention modules")
    
    # Analyze first of each type
    if spatial_attn:
        name, trace = spatial_attn[0]
        print(f"\n--- First Spatial Attention ---")
        print(f"Name: {name}")
        print(f"Input shape:  {trace.input_shapes}")
        print(f"Output shape: {trace.output_shapes}")
        
        if trace.input_shapes and len(trace.input_shapes[0]) == 3:
            B, N, C = trace.input_shapes[0]
            print(f"\nInterpretation:")
            print(f"  Batch (includes frames): {B}")
            print(f"  Sequence length (H*W): {N}")
            print(f"  Channels: {C}")
    
    if temporal_attn:
        name, trace = temporal_attn[0]
        print(f"\n--- First Temporal Attention ---")
        print(f"Name: {name}")
        print(f"Input shape:  {trace.input_shapes}")
        print(f"Output shape: {trace.output_shapes}")
        
        if trace.input_shapes and len(trace.input_shapes[0]) == 3:
            B_HW, F, C = trace.input_shapes[0]
            print(f"\nInterpretation:")
            print(f"  Batch * Spatial positions: {B_HW}")
            print(f"  Sequence length (num_frames): {F}")
            print(f"  Channels: {C}")
            print(f"\n  → Each spatial position independently attends across {F} frames")


def show_execution_order(tracer, max_items=60):
    """Show interleaving of spatial and temporal processing."""
    print("\n" + "=" * 60)
    print("EXECUTION ORDER (first 60 key modules)")
    print("=" * 60)
    
    print("\nLegend: [S]=Spatial Attention, [T]=Temporal/Motion, [R]=ResNet, [ ]=Other\n")
    
    for i, name in enumerate(tracer.execution_order[:max_items]):
        trace = tracer.traces[name]
        
        # Categorize
        if "motion_modules" in name:
            prefix = "[T]"
        elif "attentions" in name and "Attention" in trace.class_name:
            prefix = "[S]"
        elif "resnets" in name or trace.class_name == "ResnetBlock2D":
            prefix = "[R]"
        else:
            prefix = "[ ]"
        
        # Get just the last parts of the name for readability
        parts = name.split(".")
        if len(parts) > 3:
            short = ".".join(parts[-3:])
        else:
            short = name
        
        in_shape = trace.input_shapes[0] if trace.input_shapes else ()
        
        print(f"{i:3d}. {prefix} {short:<45} {trace.class_name:<25} {str(in_shape):<25}")


def save_trace_report(tracer, filepath):
    """Save full trace to file."""
    with open(filepath, 'w') as f:
        f.write("AnimateDiff UNet Forward Pass Trace\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total modules traced: {len(tracer.traces)}\n\n")
        
        f.write("EXECUTION ORDER:\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(tracer.execution_order):
            trace = tracer.traces[name]
            f.write(f"{i:4d}. [{trace.class_name}] {name}\n")
            f.write(f"       Input:  {trace.input_shapes}\n")
            f.write(f"       Output: {trace.output_shapes}\n\n")
    
    print(f"Full trace saved to: {filepath}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pipe = load_pipeline()
    
    # Trace during actual generation
    tracer = trace_during_generation(pipe)
    
    # Print summary
    tracer.print_summary(max_items=50)
    
    # Analyze attention shapes
    analyze_attention_shapes(tracer)
    
    # Show execution order  
    show_execution_order(tracer)
    
    # Save full report
    save_trace_report(tracer, OUTPUT_DIR / "unet_trace.txt")
    
    # Cleanup
    tracer.remove_hooks()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. Spatial attention input: [B*F, H*W, C]
   - Frames are batched together
   - Each frame attends spatially within itself
   
2. Temporal attention input: [B*H*W, F, C]  
   - Spatial positions are batched together
   - Each position attends across frames
   
3. This confirms: temporal consistency is enforced by having
   each pixel position "look at" its corresponding position
   in other frames, NOT by global spatiotemporal attention.
   
4. Implication for flickering: if temporal attention is weak
   or poorly trained, per-position smoothing fails, causing
   independent per-frame predictions → flicker.
""")
    
    del pipe
    torch.cuda.empty_cache()

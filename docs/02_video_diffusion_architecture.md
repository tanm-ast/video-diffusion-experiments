# Video Diffusion Architecture

## The Temporal Dimension Problem

In image diffusion (e.g., your weather prediction work), you have spatial structure. Video diffusion adds a temporal axis, and the fundamental question is: *where and how do you inject temporal awareness into a model that was designed for single images?*

## Three Main Strategies

### 1. Temporal Attention Layers (AnimateDiff approach)

Take a pretrained image model, freeze it, insert new attention layers that attend across frames.

- The image model handles per-frame quality
- The temporal layers handle consistency
- Efficient: reuse pretrained image priors

### 2. 3D Convolutions / Full Spatiotemporal (CogVideoX, research models)

Treat video as a 3D tensor from the start.

- More principled—no architectural seams
- Expensive: needs training from scratch
- Better for complex motion

### 3. Latent Interpolation + Autoregressive (older approaches)

Generate keyframes, interpolate latents.

- Simpler implementation
- Artifacts accumulate over long sequences
- Limited motion modeling

## AnimateDiff Architecture Deep Dive

### High-Level Structure

```
UNet
├── down_blocks (encoder)
│   ├── ResNet blocks (spatial)
│   ├── Spatial Attention (self-attention within frame)
│   ├── Cross Attention (text conditioning)
│   └── Motion Modules (temporal attention) ← INSERTED
├── mid_block
│   └── Same pattern
└── up_blocks (decoder)
    └── Same pattern
```

### The Motion Module

Each motion module is an `AnimateDiffTransformer3D` containing:

```
AnimateDiffTransformer3D
├── norm (GroupNorm)
├── proj_in (Linear) - project to transformer dim
├── transformer_blocks
│   └── BasicTransformerBlock
│       ├── pos_embed (SinusoidalPositionalEmbedding) - temporal position
│       ├── norm1 → attn1 (temporal self-attention)
│       ├── norm2 → attn2 (may be unused or cross-attn)
│       └── ff (feedforward)
└── proj_out (Linear) - project back
```

### How Temporal Attention Works

For a video with shape `[B, C, F, H, W]` (batch, channels, frames, height, width):

1. **Reshape for temporal attention**: `[B*H*W, F, C]`
   - Each spatial position becomes a "batch" element
   - The sequence dimension is now frames
   
2. **Apply self-attention across frames**:
   - Query, Key, Value all come from the same spatial position across different frames
   - Attention asks: "Given what this pixel looks like in frame 3, what should it look like in frame 7?"

3. **Positional embedding**: Sinusoidal encoding tells the model which frame is which

4. **Reshape back**: `[B, C, F, H, W]`

### Parameter Counts (from our inspection)

```
Total parameters: 1312.7M
├── Original SD 1.5 UNet: ~860M (frozen or finetuned)
└── Motion modules: ~450M (the temporal layers)

Temporal-related modules: 639
Spatial attention modules: 32
```

The 639 temporal modules include all subcomponents (Linear layers, norms, etc.). The key insight: temporal attention is substantial but still secondary to the spatial model.

## Why Temporal Inconsistency Happens

### Architectural Reasons

1. **Weak coupling**: Temporal attention is a learned prior over "how frames should relate," but it can be overridden by strong per-frame signals from the frozen image model.

2. **Limited receptive field**: Each spatial position only attends to the same position in other frames. Global motion or scene changes aren't directly modeled.

3. **Training data distribution**: The temporal layers learned from specific video datasets. Out-of-distribution prompts may not trigger coherent temporal priors.

### Inference-Time Reasons

1. **CFG conflict**: High classifier-free guidance strengthens per-frame prompt adherence, which can fight temporal smoothing.

2. **Noise initialization**: Independent noise per frame vs. correlated noise affects consistency.

3. **Step count**: Too few steps may not allow temporal attention to fully propagate information.

## Implications for Debugging

When you see flickering:

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| Texture shimmer | Temporal attention too weak | Check attention weights, try lower CFG |
| Color shifts | VAE decoding variance | Check latent space consistency |
| Object morphing | Spatial attention dominating | Visualize cross-frame attention |
| Motion blur artifacts | Temporal position encoding issues | Check frame ordering |

## Key Files in Diffusers

If you need to dig into the code:

```
diffusers/
├── models/
│   ├── unets/unet_motion_model.py      # AnimateDiff UNet
│   ├── attention.py                     # Attention implementations
│   └── transformers/transformer_2d.py   # Spatial transformer
├── pipelines/
│   └── animatediff/pipeline_animatediff.py
└── schedulers/
    └── scheduling_euler_discrete.py     # Our scheduler
```

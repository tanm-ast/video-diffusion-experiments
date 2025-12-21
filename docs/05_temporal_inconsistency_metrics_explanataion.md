# Temporal Consistency Metrics

Understanding and interpreting the metrics used to quantify video flickering.

---

## Overview

Temporal consistency in video means that consecutive frames change smoothly and predictably. Flickering occurs when:
- Frames change rapidly back-and-forth (oscillation)
- Motion is inconsistent between frames
- Textures/colors shift unpredictably

We measure this with multiple complementary metrics:

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| MSE | Raw pixel difference | Lower = more similar |
| PSNR | Signal quality | Higher = better |
| LPIPS | Perceptual difference | Lower = more similar |
| Flow Variance | Motion consistency | Lower = smoother |
| Warp Error | Motion accuracy | Lower = better |

---

## Metric Details

### 1. Mean Squared Error (MSE)

**What:** Average squared difference between pixel values of consecutive frames.

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (I_t[i] - I_{t+1}[i])^2$$

**Interpretation:**
- `< 0.001`: Very similar frames (static or minimal motion)
- `0.001 - 0.01`: Smooth motion
- `0.01 - 0.05`: Moderate motion or some inconsistency
- `> 0.05`: Rapid change or significant flickering

**Limitation:** Treats all pixel differences equally. A coherent motion (object moving) and random noise both increase MSE.

---

### 2. Peak Signal-to-Noise Ratio (PSNR)

**What:** Log-scale measure of reconstruction quality.

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1}{\text{MSE}}\right)$$

**Interpretation:**
- `> 40 dB`: Nearly identical frames
- `30-40 dB`: Good similarity
- `20-30 dB`: Noticeable differences
- `< 20 dB`: Significant differences

**Use:** Standard metric, easy to compare with literature.

---

### 3. LPIPS (Learned Perceptual Image Patch Similarity)

**What:** Deep learning-based perceptual similarity. Uses features from a pretrained network (AlexNet) to compare images the way humans perceive them.

**Why it matters:** Two frames can have similar MSE but very different perceptual quality. LPIPS captures:
- Texture differences
- Structural changes
- Semantic shifts

**Interpretation:**
- `< 0.1`: Very similar perceptually
- `0.1 - 0.3`: Noticeable but acceptable differences
- `> 0.3`: Significant perceptual difference

**For flickering:** High LPIPS variance across frame pairs indicates inconsistent visual changes.

---

### 4. Optical Flow Metrics

#### What is Optical Flow?

Optical flow is the pattern of apparent motion of objects between two consecutive frames. It's a 2D vector field where each vector represents the displacement (dx, dy) of a pixel from frame t to frame t+1.

```
Frame t              Frame t+1            Optical Flow
┌─────────┐         ┌─────────┐          ┌─────────┐
│    ●    │         │      ●  │          │    →→   │
│         │   →     │         │    =     │         │
│         │         │         │          │         │
└─────────┘         └─────────┘          └─────────┘
Object at (3,1)     Object at (5,1)      Flow vector (2,0)
```

#### The Brightness Constancy Assumption

Optical flow relies on a key assumption: a pixel's intensity doesn't change as it moves.

$$I(x, y, t) = I(x + dx, y + dy, t + dt)$$

Taking the Taylor expansion and simplifying:

$$\frac{\partial I}{\partial x}V_x + \frac{\partial I}{\partial y}V_y + \frac{\partial I}{\partial t} = 0$$

Or more compactly: $I_x V_x + I_y V_y + I_t = 0$

Where:
- $I_x, I_y$ = spatial gradients (how intensity changes in x, y)
- $I_t$ = temporal gradient (how intensity changes over time)
- $V_x, V_y$ = flow velocities (what we want to find)

**Problem:** One equation, two unknowns. This is the **aperture problem**—we need additional constraints.

#### Farneback Method (What We Use)

We use OpenCV's Farneback method, which approximates the neighborhood of each pixel as a polynomial:

$$f(x) \approx x^T A x + b^T x + c$$

Where A is a matrix, b is a vector, and c is a scalar, all estimated from local pixel values.

By tracking how these polynomial coefficients change between frames, we can estimate the displacement field.

**Key parameters in our code:**
```python
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2,
    None,
    pyr_scale=0.5,    # Image pyramid scale (<1 to downsample)
    levels=3,         # Number of pyramid levels
    winsize=15,       # Averaging window size
    iterations=3,     # Iterations at each pyramid level
    poly_n=5,         # Size of pixel neighborhood for polynomial
    poly_sigma=1.2,   # Gaussian std for polynomial smoothing
    flags=0
)
```

#### Sparse vs Dense Optical Flow

| Type | Method | Output | Use Case |
|------|--------|--------|----------|
| **Sparse** | Lucas-Kanade | Flow at selected feature points | Tracking, fast |
| **Dense** | Farneback, RAFT | Flow at every pixel | Full motion field, slower |

We use dense flow because we need to analyze the entire frame for flickering detection.

#### Flow Field Interpretation

The output `flow` has shape `[H, W, 2]`:
- `flow[y, x, 0]` = horizontal displacement (dx) at pixel (x, y)
- `flow[y, x, 1]` = vertical displacement (dy) at pixel (x, y)

Flow magnitude at each pixel:
$$\text{magnitude} = \sqrt{dx^2 + dy^2}$$

Flow direction at each pixel:
$$\text{angle} = \arctan2(dy, dx)$$

#### What We Compute

- **Flow magnitude mean**: Average motion (pixels/frame)
- **Flow magnitude variance**: Consistency of motion across the frame
- **Flow magnitude std**: Spread of motion speeds

**Interpretation:**

| Scenario | Mean Flow | Flow Variance |
|----------|-----------|---------------|
| Static scene | Low | Low |
| Smooth motion | Medium | Low |
| Flickering | Low-Medium | **High** |
| Chaotic motion | High | High |

**Key insight:** Flickering often appears as high flow variance with low mean flow—the pixels are changing but there's no coherent motion direction.

#### Limitations

1. **Brightness constancy violation**: Lighting changes, reflections, and transparency break the assumption
2. **Large displacements**: Basic methods fail if objects move more than a few pixels
3. **Aperture problem**: Uniform regions have ambiguous flow
4. **Occlusions**: When objects disappear/appear, flow is undefined

---

### 5. Warp Error

**What:** Warp frame t using optical flow, compare to frame t+1.

$$\text{Warp Error} = \text{MSE}(\text{warp}(I_t, \text{flow}), I_{t+1})$$

If motion is perfectly captured by flow, warped frame should match the next frame exactly.

#### How Warping Works

Given optical flow, we can "move" pixels from frame t to predict frame t+1:

```
Frame t               Flow                 Warped (prediction of t+1)
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ R G B . .   │      │ → → . . .   │      │ . R G B .   │
│ . . . . .   │  +   │ . . . . .   │  =   │ . . . . .   │
│ . . . . .   │      │ . . . . .   │      │ . . . . .   │
└─────────────┘      └─────────────┘      └─────────────┘

Then compare warped frame to actual frame t+1.
```

#### Backward Warping (What We Use)

There are two approaches:

**Forward warping:** Push each source pixel to its destination
- Problem: Multiple pixels may land on the same destination (collision)
- Problem: Some destinations may have no source (holes)

**Backward warping:** For each destination pixel, ask "where did you come from?"
- Sample from the source at that location
- Use bilinear interpolation if sampling between pixels
- No holes, no collisions

We use backward warping with PyTorch's `grid_sample`:

```python
# For each output position, compute where to sample from
sample_x = grid_x + flow[..., 0]  # Original position + flow
sample_y = grid_y + flow[..., 1]

# grid_sample expects normalized coordinates [-1, 1]
# -1 = left/top edge, +1 = right/bottom edge
sample_x = 2 * sample_x / (W - 1) - 1
sample_y = 2 * sample_y / (H - 1) - 1

# Warp using bilinear interpolation
warped = F.grid_sample(frame, grid, mode='bilinear')
```

#### Interpretation

- Low warp error: Motion is consistent and predictable
- High warp error: Motion doesn't follow expected patterns

**Causes of high warp error:**
- Occlusions (objects appearing/disappearing)
- Non-rigid motion (deformation)
- Flickering (random per-frame noise that can't be explained by motion)
- Flow estimation failures

---

### 6. Temporal Consistency Score (Composite)

**What:** Single number combining multiple metrics.

$$\text{Score} = w_1 \cdot \text{Var}(\text{MSE}) + w_2 \cdot \text{Mean}(\text{MSE}) + w_3 \cdot \text{Mean}(\text{LPIPS}) + w_4 \cdot \text{Var}(\text{LPIPS})$$

**Interpretation:**
- Lower = more temporally consistent
- Higher = more flickering/inconsistency

**Components:**
- MSE variance: Catches inconsistent frame-to-frame changes
- MSE mean: Overall change magnitude
- LPIPS mean/variance: Perceptual consistency

---

### 7. Flicker Index (Second-Order)

**What:** Detects oscillation (frame going A→B→A).

$$\text{Flicker} = \frac{1}{T-2} \sum_{t=0}^{T-3} |I_t - 2 \cdot I_{t+1} + I_{t+2}|$$

This is the discrete second derivative. High values mean frames are oscillating rather than progressing smoothly.

**Interpretation:**
- Low: Smooth progression
- High: Back-and-forth flickering

---

## Using Metrics for Debugging

### Diagnose the Type of Problem

| Symptom | Likely Cause |
|---------|--------------|
| High MSE variance, low mean | Flickering (rapid small changes) |
| High MSE mean, low variance | Fast but consistent motion |
| High LPIPS, low MSE | Texture/style shifts |
| High flow variance | Inconsistent motion direction |
| High warp error | Occlusions or motion discontinuities |

### Compare Interventions

When testing fixes:

1. Generate baseline video
2. Measure metrics
3. Apply intervention (e.g., lower CFG, add temporal blending)
4. Generate new video with same seed
5. Measure metrics
6. Compare: Did consistency score decrease?

### What Good Numbers Look Like

For a 16-frame, 512x512 video with moderate motion:

| Metric | Poor | Acceptable | Good |
|--------|------|------------|------|
| Mean MSE | >0.03 | 0.01-0.03 | <0.01 |
| MSE Std | >0.01 | 0.005-0.01 | <0.005 |
| Mean LPIPS | >0.2 | 0.1-0.2 | <0.1 |
| Consistency Score | >5 | 2-5 | <2 |

---

## Limitations

1. **Motion vs. flickering ambiguity**: Fast legitimate motion and flickering both increase metrics. Use flow direction consistency to disambiguate.

2. **Content-dependent baselines**: A video of a static scene should have much lower metrics than an action scene. Compare like with like.

3. **Perceptual nuance**: Some "flickering" is actually fine (e.g., water reflections, fire). Metrics can't capture semantic appropriateness.

4. **Compression artifacts**: If frames are JPEG compressed, artifacts add noise to metrics.

---

## References

- LPIPS: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
- Optical Flow: Farneback, "Two-Frame Motion Estimation Based on Polynomial Expansion" (2003)
- Video Quality: Wang et al., "Video Quality Assessment" (comprehensive survey)
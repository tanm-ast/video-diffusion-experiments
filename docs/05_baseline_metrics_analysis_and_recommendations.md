# Baseline Metrics Analysis and Recommendations

Analysis of temporal consistency metrics from baseline video generation experiments.

---

## Baseline Results Summary

| Video | MSE Mean | MSE Std | LPIPS | Flow Mag | Flow Var | Consistency Score |
|-------|----------|---------|-------|----------|----------|-------------------|
| portrait | 0.0028 | 0.0022 | 0.052 | 0.37 | 0.020 | **3.05** ✓ |
| landscape | 0.0072 | 0.0042 | 0.071 | 0.23 | 0.002 | **4.72** ✓ |
| woman_waving | 0.0102 | 0.0034 | 0.181 | 2.85 | 0.345 | 10.60 |
| mig21_missile | 0.0113 | 0.0040 | 0.207 | 3.36 | 1.338 | 13.91 |
| corgi_beach | 0.0184 | 0.0049 | 0.292 | 5.17 | 2.199 | 17.10 |
| birds_flying | 0.0211 | 0.0058 | **0.380** | 3.95 | 0.317 | **22.80** ✗ |

Baseline configuration:
- `guidance_scale`: 7.5
- `num_inference_steps`: 25
- `num_frames`: 16
- Resolution: 512×512

---

## Key Observations

### 1. Static Scenes Perform Best

**Portrait** and **landscape** achieve the best consistency scores. Minimal motion means temporal attention has an easier task—pixels don't move much between frames, so per-position attention across frames works well.

This establishes the "easy case" baseline and confirms the metrics behave as expected.

### 2. birds_flying Shows Texture Flickering

Despite moderate flow magnitude (3.95 px/frame), birds_flying has the highest LPIPS (0.380)—significantly worse than corgi_beach (0.292) which has higher motion.

**Interpretation:** The birds aren't moving fast, but their *appearance* changes dramatically frame-to-frame. This indicates texture flickering: bird shapes, feathers, or sky texture regenerating inconsistently despite minimal spatial displacement.

This is the signature of temporal attention failure on fine-grained textures.

### 3. High Flow Variance Correlates with Scene Complexity

| Video | Flow Variance | Scene Characteristics |
|-------|---------------|----------------------|
| landscape | 0.002 | Static scene |
| portrait | 0.020 | Subtle facial micro-movements |
| birds_flying | 0.317 | Multiple independent objects |
| woman_waving | 0.345 | Articulated hand motion |
| mig21_missile | 1.338 | Fast object + static background |
| corgi_beach | **2.199** | Multiple motion sources (dog, waves, camera) |

High flow variance indicates the model is generating inconsistent motion directions—different parts of the frame moving in incompatible ways, or motion that changes erratically between frames.

### 4. MSE Std/Mean Ratio Reveals Hidden Flickering

| Video | MSE Std/Mean | Interpretation |
|-------|--------------|----------------|
| portrait | **0.79** | High ratio despite low motion |
| landscape | 0.58 | Moderate |
| woman_waving | 0.33 | Consistent changes |
| mig21_missile | 0.35 | Consistent changes |
| corgi_beach | 0.27 | Consistent large changes |
| birds_flying | 0.27 | Consistent large changes |

**Notable finding:** Portrait has the highest ratio (0.79). Despite very low absolute MSE, the frame-to-frame changes are inconsistent. This suggests subtle face flickering—imperceptible in casual viewing but measurable. Likely caused by fine facial features (skin texture, eye details) regenerating slightly differently each frame.

---

## Per-Video Diagnosis

### birds_flying (Consistency Score: 22.80)

**Primary issue:** Texture flickering on small objects

**Evidence:**
- Very high LPIPS (0.38) indicates perceptual inconsistency
- Moderate flow magnitude (3.95) rules out motion blur as the cause
- Multiple small objects (birds) provide minimal spatial context for temporal attention

**Likely cause:** Each bird occupies only a few pixels. Per-position temporal attention has insufficient context to maintain consistency. The model essentially regenerates bird textures each frame.

**Visual prediction:** Birds appear to shimmer, morph, or have unstable feather textures.

---

### corgi_beach (Consistency Score: 17.10)

**Primary issue:** Multiple competing motion sources

**Evidence:**
- Highest flow variance (2.199) indicates inconsistent motion
- High MSE mean (0.018) shows large frame-to-frame changes
- High LPIPS (0.292) confirms perceptual instability

**Likely cause:** The scene contains multiple independent motion sources: walking dog, ocean waves, possibly camera movement. Temporal attention cannot simultaneously track all of these, leading to compromises where some elements flicker.

**Visual prediction:** Water/wave textures flicker, dog outline may shimmer, beach texture unstable.

---

### mig21_missile (Consistency Score: 13.91)

**Primary issue:** Fast-moving object challenges temporal tracking

**Evidence:**
- High flow variance (1.338)
- Moderate LPIPS (0.207)
- Fast-moving small object (missile) against relatively static background

**Likely cause:** The missile moves many pixels per frame, exceeding what temporal attention can smoothly track. The background is easier to keep consistent, but the missile and its trail likely flicker.

**Visual prediction:** Missile edges shimmer, exhaust trail inconsistent, background relatively stable.

---

### woman_waving (Consistency Score: 10.60)

**Primary issue:** Complex articulated motion

**Evidence:**
- Moderate metrics across the board
- Flow variance (0.345) suggests localized motion complexity
- LPIPS (0.181) indicates some perceptual inconsistency

**Likely cause:** Hand and arm articulation is inherently complex—fingers appear/disappear, wrist rotates. The rest of the figure (face, torso) should be relatively stable.

**Visual prediction:** Hand/arm edges flicker, possible finger artifacts, face and body relatively stable.

---

### portrait (Consistency Score: 3.05)

**Primary issue:** Subtle texture flickering despite static pose

**Evidence:**
- Very low MSE mean (0.0028) but high MSE std/mean ratio (0.79)
- Low LPIPS (0.052) suggests changes are below perceptual threshold
- Minimal flow (0.37 px/frame)

**Likely cause:** Fine facial details (skin pores, eye reflections, hair strands) regenerate with slight variations each frame. Not perceptible in normal viewing but measurable.

**Visual prediction:** Imperceptible flickering; may become visible if video is slowed down or examined frame-by-frame.

---

### landscape (Consistency Score: 4.72)

**Status:** Acceptable performance

**Evidence:**
- Low metrics across all measures
- Minimal flow (0.23 px/frame)
- Low LPIPS (0.071)

**Assessment:** Static scenes with no expected motion perform well. This represents the best-case scenario for current temporal attention mechanisms.

---

## Root Cause Analysis

The metrics reveal a consistent pattern: **temporal attention fails when the task exceeds its architectural capacity**.

Specifically:

| Failure Mode | Cause | Affected Videos |
|--------------|-------|-----------------|
| Fine texture flickering | Per-position attention lacks spatial context | birds_flying, portrait |
| Multi-source motion | Cannot track independent objects | corgi_beach |
| Fast motion | Displacement exceeds attention receptive field | mig21_missile |
| Articulated motion | Complex topology changes | woman_waving |

The common thread: AnimateDiff's temporal attention operates per-spatial-position. Each position can only see itself across frames. This is fundamentally limited for:
- Small objects (insufficient context)
- Large displacements (position correspondence breaks)
- Multiple objects (no global coordination)

---

## Recommended Hyperparameter Changes

### Global Improvements

```python
# Baseline (current)
baseline_config = {
    "guidance_scale": 7.5,
    "num_inference_steps": 25,
}

# Improved default
improved_config = {
    "guidance_scale": 6.0,      # Lower CFG reduces per-frame independence
    "num_inference_steps": 30,  # More steps for better convergence
}

# Enhanced negative prompt
negative_prompt = (
    "bad quality, blurry, distorted, ugly, deformed, "
    "flickering, morphing, inconsistent, jittery, "
    "changing appearance, unstable"
)
```

**Rationale for lower CFG:** Classifier-free guidance at 7.5 strongly enforces per-frame prompt adherence, which competes with temporal attention's smoothing effect. Reducing to 6.0 shifts the balance toward temporal consistency.

---

### Per-Video Recommendations

#### birds_flying

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **5.0** | Significantly lower to prioritize temporal smoothing |
| num_inference_steps | 25 | **35** | More steps for complex multi-object scene |
| prompt addition | - | "smooth motion, consistent shapes" | Explicit consistency guidance |
| negative addition | - | "morphing birds, changing shapes, jittery feathers" | Target specific failure mode |

---

#### corgi_beach

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **5.5** | Lower to reduce motion source competition |
| num_inference_steps | 25 | **30** | Moderate increase |
| prompt addition | - | "steady camera, smooth motion" | Reduce camera motion |
| negative addition | - | "flickering water, unstable waves, shaky" | Target water texture |

---

#### mig21_missile

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **6.0** | Moderate reduction |
| num_inference_steps | 25 | **40** | More steps for fast motion resolution |
| prompt addition | - | "smooth motion blur, cinematic, steady shot" | Motion blur hides flickering |
| negative addition | - | "jittery, teleporting, inconsistent trail" | Target missile artifacts |

---

#### woman_waving

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **6.5** | Slight reduction |
| num_inference_steps | 25 | **30** | Moderate increase |
| prompt addition | - | "smooth natural motion" | General smoothness |
| negative addition | - | "flickering hands, morphing fingers, distorted hands" | Target hand artifacts |

---

#### portrait

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **7.0** | Minimal change—static scene tolerates higher CFG |
| num_inference_steps | 25 | **35** | More steps for fine facial detail |
| prompt addition | - | "static pose, consistent lighting" | Reinforce stillness |
| negative addition | - | "changing expression, morphing face" | Target facial flickering |

---

#### landscape

| Parameter | Baseline | Recommended | Rationale |
|-----------|----------|-------------|-----------|
| guidance_scale | 7.5 | **7.0** | Minimal change—already performs well |
| num_inference_steps | 25 | **25** | No change needed |
| prompt addition | - | "still water, serene" | Prevent motion generation |
| negative addition | - | "flickering water, rippling, windy" | Prevent dynamic elements |

---

## Experimental Priority

For limited compute budget, test in this order:

| Priority | Change | Expected Impact | Compute Cost |
|----------|--------|-----------------|--------------|
| 1 | Lower CFG to 5.5-6.0 | High | Low |
| 2 | Add consistency terms to negative prompt | Medium | None |
| 3 | Increase steps to 30-35 | Medium | Medium |
| 4 | Modify positive prompt for smoothness | Low-Medium | None |

The highest-impact intervention is reducing guidance_scale. This directly addresses the architectural tension between per-frame prompt adherence and temporal smoothing.

---

## Expected Outcomes

Based on the analysis, the recommended changes should produce:

| Video | Current Score | Expected Score | Primary Improvement |
|-------|---------------|----------------|---------------------|
| birds_flying | 22.80 | 15-18 | Reduced texture flickering |
| corgi_beach | 17.10 | 12-14 | More stable water/background |
| mig21_missile | 13.91 | 10-12 | Smoother missile motion |
| woman_waving | 10.60 | 8-10 | Reduced hand artifacts |
| portrait | 3.05 | 2.5-3.0 | Minor improvement |
| landscape | 4.72 | 4.0-4.5 | Minor improvement |

These are estimates. Actual results require experimental validation.

---

## Limitations of Hyperparameter Tuning

Hyperparameter changes can mitigate but not fully solve temporal inconsistency. The fundamental limitation is architectural:

1. **Per-position temporal attention** cannot model global motion
2. **No explicit motion representation** (optical flow, trajectories)
3. **Fixed 16-frame context** limits long-range consistency

For significant improvements beyond hyperparameter tuning, architectural modifications or post-processing would be required.

---

*Analysis based on AnimateDiff v1.5.2 with Stable Diffusion 1.5 backbone.*
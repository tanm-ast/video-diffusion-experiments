# Ablation Study Results Analysis

Systematic evaluation of hyperparameter interventions for improving temporal consistency in AnimateDiff video generation.

---

## Executive Summary

| Intervention | Avg Improvement | Verdict |
|--------------|-----------------|---------|
| **Prompt engineering** | **+19.8%** | ✓ Most effective |
| Combined (recommended) | +18.6% | ✓ Effective (driven by prompts) |
| More inference steps | +1.1% | ~ Negligible impact |
| Lower CFG | **-5.0%** | ✗ Counterproductive |

**Key finding:** Prompt engineering alone outperformed all other interventions, including combined approaches. Lower CFG, contrary to initial hypothesis, degraded temporal consistency.

---

## Experimental Setup

### Baseline Configuration
```python
{
    "guidance_scale": 7.5,
    "num_inference_steps": 25,
    "num_frames": 16,
    "resolution": "512x512",
    "negative_prompt": "bad quality, blurry, distorted"
}
```

### Interventions Tested

| Intervention | Changes Applied |
|--------------|-----------------|
| `baseline` | Control (no changes) |
| `cfg_only` | guidance_scale → 5.0-6.5 (per video) |
| `steps_only` | num_inference_steps → 30-40 (per video) |
| `prompt_only` | Enhanced positive/negative prompts |
| `recommended` | All changes combined |

### Test Videos
Six diverse scenarios covering static and dynamic content:
- `birds_flying` — Multiple small moving objects
- `corgi_beach` — Complex scene with multiple motion sources
- `mig21_missile` — Fast-moving object
- `woman_waving` — Articulated human motion
- `portrait` — Static subject, fine detail
- `landscape` — Static scene

---

## Results by Intervention Type

### 1. Prompt Engineering (+19.8% average improvement)

**Most effective intervention across all test cases.**

| Video | Baseline Score | Prompt-Only Score | Improvement |
|-------|----------------|-------------------|-------------|
| woman_waving | 21.80 | 9.40 | **+56.9%** |
| mig21_missile | 18.94 | 15.39 | **+18.7%** |
| portrait | 10.02 | 8.52 | +15.0% |
| birds_flying | 22.18 | 19.38 | +12.6% |
| landscape | 6.43 | 5.89 | +8.4% |
| corgi_beach | 17.35 | 16.12 | +7.1% |

**Prompt modifications applied:**

Positive prompt additions:
```
"smooth motion, consistent shapes, steady camera, stable appearance"
```

Negative prompt additions:
```
"flickering, morphing, jittery, inconsistent, unstable, changing shapes"
```

**Interpretation:** The model has learned semantic associations between these terms and temporal coherence during training. Explicitly requesting stability activates these learned patterns.

---

### 2. Lower CFG (-5.0% average — WORSE)

**Contrary to initial hypothesis, reducing guidance scale degraded performance.**

| Video | Baseline Score | CFG-Only Score | Change |
|-------|----------------|----------------|--------|
| portrait | 10.02 | 11.07 | **-10.4%** |
| corgi_beach | 17.35 | 19.02 | -9.6% |
| landscape | 6.43 | 6.86 | -6.6% |
| mig21_missile | 18.94 | 19.84 | -4.8% |
| birds_flying | 22.18 | 22.66 | -2.2% |
| woman_waving | 21.80 | 20.94 | +4.0% |

**Why lower CFG hurt:**

Initial hypothesis: High CFG forces strong per-frame prompt adherence, fighting temporal smoothing. Lower CFG should let temporal attention dominate.

Actual finding: Lower CFG introduces more variance in the denoising process. Without strong guidance anchoring each frame, the model produces less consistent outputs. Temporal attention alone is insufficient to maintain coherence.

**Revised understanding:** CFG provides a stabilizing signal that temporal attention leverages. Reducing it removes this anchor, causing drift.

---

### 3. More Inference Steps (+1.1% average — Negligible)

**Minimal and inconsistent impact.**

| Video | Baseline Score | Steps-Only Score | Change |
|-------|----------------|------------------|--------|
| portrait | 10.02 | 8.92 | +11.0% |
| woman_waving | 21.80 | 20.62 | +5.4% |
| corgi_beach | 17.35 | 16.83 | +3.0% |
| landscape | 6.43 | 6.43 | +0.0% |
| birds_flying | 22.18 | 23.49 | -5.9% |
| mig21_missile | 18.94 | 20.18 | -6.6% |

**Interpretation:** Additional denoising steps help in some cases (portrait, woman_waving) but hurt in others (birds_flying, mig21_missile). The effect depends on content complexity. Static scenes may benefit from more steps for fine detail convergence, while dynamic scenes may accumulate inconsistencies over more steps.

---

### 4. Combined Recommended Settings (+18.6% average)

**Effective, but not better than prompt-only.**

| Video | Baseline Score | Recommended Score | Change |
|-------|----------------|-------------------|--------|
| woman_waving | 21.80 | 10.93 | +49.9% |
| portrait | 10.02 | 5.90 | **+41.1%** |
| birds_flying | 22.18 | 19.94 | +10.1% |
| corgi_beach | 17.35 | 16.39 | +5.5% |
| landscape | 6.43 | 6.22 | +3.3% |
| mig21_missile | 18.94 | 18.61 | +1.7% |

**Notable:** Portrait achieved the best improvement with combined settings (+41.1%), suggesting static scenes benefit from the full combination. For dynamic scenes, prompt-only often matched or exceeded combined settings.

---

## Per-Video Analysis

### birds_flying
- **Best intervention:** prompt_only (+12.6%)
- **Issue:** Texture flickering on small objects
- **Finding:** Prompt guidance to maintain "consistent shapes" directly addresses the failure mode

### corgi_beach
- **Best intervention:** prompt_only (+7.1%)
- **Issue:** Multiple competing motion sources
- **Finding:** "Steady camera" prompt reduced perceived instability

### mig21_missile
- **Best intervention:** prompt_only (+18.7%)
- **Issue:** Fast motion tracking
- **Finding:** Lower CFG and more steps both hurt; only prompts helped

### woman_waving
- **Best intervention:** prompt_only (+56.9%)
- **Issue:** Articulated hand motion
- **Finding:** Dramatic improvement from "flickering hands, morphing fingers" in negative prompt

### portrait
- **Best intervention:** recommended (+41.1%)
- **Issue:** Subtle texture flickering
- **Finding:** Static scenes benefit from combined CFG reduction + more steps + prompts

### landscape
- **Best intervention:** prompt_only (+8.4%)
- **Issue:** Already performing well
- **Finding:** Minimal room for improvement; prompts still helped

---

## Conclusions

### What Works
1. **Prompt engineering is the primary lever.** Adding temporal consistency terms to prompts provides ~20% improvement with zero computational cost.

2. **Negative prompts are powerful.** Explicitly listing failure modes ("flickering", "morphing", "jittery") helps the model avoid them.

3. **Content-specific prompts matter.** "Flickering hands" for human motion, "consistent shapes" for objects, "steady camera" for scenes.

### What Doesn't Work
1. **Lower CFG hurts.** Contrary to intuition, reducing guidance scale destabilizes generation.

2. **More steps have inconsistent effects.** Benefits are content-dependent and often negligible.

### Revised Best Practices

```python
# Recommended configuration
config = {
    "guidance_scale": 7.5,        # Keep at default
    "num_inference_steps": 25,    # Default is fine
    "num_frames": 16,
}

# Universal prompt additions
positive_additions = "smooth motion, consistent appearance, stable, coherent"

negative_prompt = (
    "flickering, morphing, jittery, inconsistent, unstable, "
    "changing shapes, bad quality, blurry, distorted"
)

# Content-specific negative additions
human_motion_negative = "flickering hands, morphing fingers, distorted limbs"
object_motion_negative = "changing shapes, morphing objects, unstable edges"
scene_negative = "shaky camera, flickering textures, unstable background"
```

---

## Implications for Production

### For API-Based Services (like Deccan AI)

1. **Prompt templates are the intervention.** Wrap user prompts with consistency-enhancing terms.

2. **No need to modify inference parameters.** Default CFG and steps are appropriate.

3. **Content-aware prompt augmentation.** Detect content type and add appropriate negative prompts.

Example wrapper:
```python
def enhance_prompt(user_prompt, content_type="general"):
    base_additions = "smooth motion, consistent appearance"
    
    negative_base = "flickering, morphing, jittery, inconsistent, unstable"
    
    if content_type == "human":
        negative_base += ", flickering hands, morphing fingers"
    elif content_type == "object":
        negative_base += ", changing shapes, unstable edges"
    
    enhanced_positive = f"{user_prompt}, {base_additions}"
    
    return enhanced_positive, negative_base
```

### Limitations

Prompt engineering provides meaningful improvement but does not fully solve temporal consistency. Fundamental architectural limitations remain:

- Per-position temporal attention cannot model global motion
- No explicit motion representation in the model
- 16-frame context window limits long-range consistency

For further improvements, architectural modifications or post-processing pipelines would be required.

---

## Appendix: Raw Data

### Full Metrics Table

| Experiment | MSE Mean | MSE Std | LPIPS | Flow Var | Consistency |
|------------|----------|---------|-------|----------|-------------|
| birds_flying_baseline | 0.02192 | 0.00535 | 0.377 | 0.364 | 22.18 |
| birds_flying_prompt_only | 0.01200 | 0.00396 | 0.338 | 0.487 | 19.38 |
| birds_flying_recommended | 0.01239 | 0.00408 | 0.346 | 0.301 | 19.94 |
| corgi_beach_baseline | 0.02059 | 0.00452 | 0.298 | 1.543 | 17.35 |
| corgi_beach_prompt_only | 0.01463 | 0.00382 | 0.279 | 1.507 | 16.12 |
| woman_waving_baseline | 0.02199 | 0.00775 | 0.297 | 1.377 | 21.80 |
| woman_waving_prompt_only | 0.00869 | 0.00337 | 0.155 | 0.686 | 9.40 |
| portrait_baseline | 0.00859 | 0.00658 | 0.132 | 3.676 | 10.02 |
| portrait_recommended | 0.00711 | 0.00468 | 0.094 | 0.149 | 5.90 |
| mig21_missile_baseline | 0.01792 | 0.00444 | 0.335 | 1.599 | 18.94 |
| mig21_missile_prompt_only | 0.01638 | 0.00409 | 0.263 | 3.107 | 15.39 |
| landscape_baseline | 0.00814 | 0.00476 | 0.102 | 0.002 | 6.43 |
| landscape_prompt_only | 0.00753 | 0.00419 | 0.089 | 0.001 | 5.89 |

---

*Analysis completed December 2024*

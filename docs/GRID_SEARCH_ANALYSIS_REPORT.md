# AnimateDiff Temporal Consistency: Grid Search Ablation Study

**Project:** Video Diffusion Temporal Consistency Research  
**Date:** December 2025

---

## Executive Summary

This study systematically tested hyperparameter configurations for AnimateDiff to identify optimal settings for temporal consistency. **Our initial hypotheses were proven wrong by the data:**

| Hypothesis | Expected | Actual Finding |
|------------|----------|----------------|
| Lower CFG improves consistency | CFG 5-6 best | **CFG 9.0 best** (7/8 metrics agree) |
| More steps improves quality | Steps 40-50 best | **Steps 15 best** (8/8 metrics agree for some videos) |
| Enhanced prompts always help | Universal improvement | **Content-dependent** (helps 3/6, hurts 2/6) |

---

## Visual Evidence: Key Findings

### Finding 1: Higher CFG Produces Better Temporal Consistency

**Portrait - Static Content**

| CFG 5.0 (Worse) | CFG 9.0 (Better) |
|-----------------|------------------|
| ![CFG 5.0](assets/portrait_cfg5.0_steps25.gif) | ![CFG 9.0](assets/portrait_cfg9.0_steps25.gif) |
| More flickering, unstable features | Stable, consistent frames |

**Birds Flying - Dynamic Content**

| CFG 5.0 (Worse) | CFG 9.0 (Better) |
|-----------------|------------------|
| ![CFG 5.0](assets/birds_flying_cfg5.0_steps25.gif) | ![CFG 9.0](assets/birds_flying_cfg9.0_steps25.gif) |
| Morphing shapes, erratic motion | Consistent bird shapes |

---

### Finding 2: Fewer Inference Steps Produce Better Results

**Landscape - Static Scene**

| Steps 15 (Better) | Steps 50 (Worse) |
|-------------------|------------------|
| ![Steps 15](assets/landscape_cfg7.5_steps15.gif) | ![Steps 50](assets/landscape_cfg7.5_steps50.gif) |
| Stable water reflection | Flickering, rippling artifacts |

**Birds Flying - Dynamic Scene**

| Steps 15 (Better) | Steps 50 (Worse) |
|-------------------|------------------|
| ![Steps 15](assets/birds_flying_cfg7.5_steps15.gif) | ![Steps 50](assets/birds_flying_cfg7.5_steps50.gif) |
| Smooth flight path | Erratic motion, more variance |

---

### Finding 3: Prompt Engineering is Content-Dependent

**✅ HELPS: Woman Waving (Natural Motion)**

| Baseline Prompt | Enhanced Prompt |
|-----------------|-----------------|
| ![Baseline](assets/woman_waving_cfg7.5_steps25_prompt_baseline.gif) | ![Enhanced](assets/woman_waving_cfg7.5_steps25_prompt_enhanced.gif) |
| Some hand flickering | Smooth, natural motion |

*Enhanced prompt adds: "smooth natural motion" and negative "flickering hands, morphing fingers"*

**❌ HURTS: Portrait (Static Content)**

| Baseline Prompt | Enhanced Prompt |
|-----------------|-----------------|
| ![Baseline](assets/portrait_cfg7.5_steps25_prompt_baseline.gif) | ![Enhanced](assets/portrait_cfg7.5_steps25_prompt_enhanced.gif) |
| Stable, minimal motion | **Introduced unwanted motion** |

*Enhanced prompt backfired: adding motion-related terms to static content causes instability*

**❌ HURTS: MiG-21 Missile (Fast Action)**

| Baseline Prompt | Enhanced Prompt |
|-----------------|-----------------|
| ![Baseline](assets/mig21_missile_cfg7.5_steps25_prompt_baseline.gif) | ![Enhanced](assets/mig21_missile_cfg7.5_steps25_prompt_enhanced.gif) |
| Acceptable motion blur | **Increased flickering and artifacts** |

*Enhanced prompt backfired: "smooth motion blur" term conflicted with fast action content*

---

## Methodology

### Experimental Design

**Principle:** Vary one parameter at a time to isolate effects.

| Phase | Parameter | Values Tested | Fixed Parameters |
|-------|-----------|---------------|------------------|
| 1. CFG Ablation | Guidance Scale | 5.0, 6.0, 7.0, 7.5, 8.0, 9.0 | Steps=25, Baseline prompts |
| 2. Steps Ablation | Inference Steps | 15, 20, 25, 30, 40, 50 | CFG=7.5, Baseline prompts |
| 3. Prompt Ablation | Prompt Type | Baseline, Enhanced | CFG=7.5, Steps=25 |

**Test Videos (6 content types):**
- `birds_flying` - Dynamic natural motion
- `corgi_beach` - Animal motion with background
- `woman_waving` - Human gesture
- `portrait` - Static human subject
- `landscape` - Static natural scene
- `mig21_missile` - Fast mechanical action

**Total Experiments:** 78 unique configurations

### Metrics Used (8 total, lower is better for all)

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| `mean_mse` | Pixel-level frame difference | Overall change magnitude |
| `mean_lpips` | Perceptual difference (AlexNet) | Human-perceived change |
| `mean_flow_magnitude` | Amount of motion (px/frame) | Motion intensity |
| `flow_magnitude_variance` | Motion consistency | Erratic motion detection |
| `mean_warp_error` | Motion prediction error | Unpredictable motion |
| `warp_error_variance` | Prediction consistency | Stability of predictions |
| `flicker_index` | Second-order temporal difference | Direct flicker detection |
| `temporal_consistency_score` | Composite score | Overall quality |

---

## Results

### 1. CFG Sweep Analysis

#### Win Counts: Which CFG Value is Optimal?

| Metric | CFG 5.0 | CFG 6.0 | CFG 7.0 | CFG 7.5 | CFG 8.0 | CFG 9.0 | **Winner** |
|--------|---------|---------|---------|---------|---------|---------|------------|
| MSE | 0 | 0 | 1 | 0 | 1 | **4** | CFG 9.0 |
| LPIPS | 0 | 0 | 1 | 0 | 0 | **5** | CFG 9.0 |
| Flow Magnitude | 1 | 0 | 1 | 0 | 0 | **4** | CFG 9.0 |
| Flow Variance | 1 | 0 | 1 | 1 | 1 | **2** | CFG 9.0 |
| Warp Error | 0 | 0 | 1 | **2** | 1 | 2 | CFG 7.5/9.0 |
| Warp Variance | 0 | 0 | 0 | 0 | 0 | **6** | CFG 9.0 |
| Flicker Index | 0 | 0 | 1 | 0 | 0 | **5** | CFG 9.0 |
| Consistency | 0 | 0 | 1 | 0 | 0 | **5** | CFG 9.0 |

**Conclusion:** CFG 9.0 wins 7 out of 8 metrics. Higher guidance scale produces more stable outputs.

#### Trend Analysis

| Metric | Higher CFG Better | Lower CFG Better | Mixed |
|--------|-------------------|------------------|-------|
| MSE | 4 videos | 0 videos | 2 videos |
| LPIPS | **6 videos** | 0 videos | 0 videos |
| Flow Magnitude | 4 videos | 1 video | 1 video |
| Warp Variance | **6 videos** | 0 videos | 0 videos |
| Flicker Index | 4 videos | 0 videos | 2 videos |

**Exception:** `mig21_missile` (fast action) prefers CFG 7.0 - lower guidance allows more motion blur.

---

### 2. Steps Sweep Analysis

#### Win Counts: Which Steps Value is Optimal?

| Metric | Steps 15 | Steps 20 | Steps 25 | Steps 30 | Steps 40 | Steps 50 | **Winner** |
|--------|----------|----------|----------|----------|----------|----------|------------|
| MSE | **3** | 1 | 0 | 0 | 1 | 1 | Steps 15 |
| LPIPS | **3** | 1 | 0 | 0 | 0 | 2 | Steps 15 |
| Flow Magnitude | **3** | 0 | 0 | 1 | 1 | 1 | Steps 15 |
| Flow Variance | **3** | 1 | 0 | 1 | 1 | 0 | Steps 15 |
| Warp Error | **3** | 1 | 0 | 0 | 1 | 1 | Steps 15 |
| Warp Variance | **4** | 1 | 0 | 1 | 0 | 0 | Steps 15 |
| Flicker Index | **4** | 1 | 0 | 0 | 1 | 0 | Steps 15 |
| Consistency | **3** | 1 | 0 | 0 | 0 | 2 | Steps 15 |

**Conclusion:** Steps 15 wins across all metrics. Fewer inference steps = less opportunity for frame-to-frame drift.

#### Trend Analysis

| Metric | Fewer Steps Better | More Steps Better | Mixed |
|--------|-------------------|-------------------|-------|
| MSE | 4 videos | 1 video | 1 video |
| LPIPS | 3 videos | 1 video | 2 videos |
| Warp Variance | **5 videos** | 1 video | 0 videos |
| Flicker Index | 4 videos | 0 videos | 2 videos |

**Exceptions:** `portrait` prefers 40-50 steps, `woman_waving` prefers 50 steps for some metrics.

---

### 3. Prompt Engineering Analysis

#### Per-Video Impact (% Change from Baseline, Positive = Improved)

| Video | MSE | LPIPS | Flow Mag | Flow Var | Warp Err | Flicker | **Verdict** |
|-------|-----|-------|----------|----------|----------|---------|-------------|
| birds_flying | +42% ✓ | -1% ~ | +42% ✓ | +29% ✓ | +43% ✓ | +19% ✓ | **Helps** |
| corgi_beach | +28% ✓ | +13% ✓ | -9% ✗ | +15% ✓ | +26% ✓ | +16% ✓ | **Helps** |
| woman_waving | +46% ✓ | +32% ✓ | +75% ✓ | +99% ✓ | +51% ✓ | +17% ✓ | **Helps** |
| landscape | -1% ~ | 0% ~ | +22% ✓ | +27% ✓ | 0% ~ | -2% ~ | Neutral |
| mig21_missile | -83% ✗ | -2% ~ | -77% ✗ | -226% ✗ | -77% ✗ | -37% ✗ | **Hurts** |
| portrait | -168% ✗ | -40% ✗ | -174% ✗ | -10624% ✗ | -209% ✗ | -51% ✗ | **Hurts** |

*✓ = Improved (>5%), ✗ = Worse (<-5%), ~ = Neutral*

#### Summary Statistics

| Metric | Avg Improvement | Wins | Losses | Verdict |
|--------|-----------------|------|--------|---------|
| MSE | -22.6% | 3 | 2 | Mixed |
| LPIPS | +0.6% | 2 | 1 | Mixed |
| Flow Variance | -1780% | 4 | 2 | Mixed |
| Flicker Index | -6.6% | 3 | 2 | Mixed |

**Key Insight:** The massive negative averages are driven by portrait and mig21_missile catastrophic failures. Enhanced prompts are powerful but must be used selectively.

---

### 4. Metric Agreement Analysis

How well do the 8 metrics agree on optimal values?

#### CFG Agreement

| Video | Unique Optimal CFGs | Agreement Score | Most Common CFG |
|-------|---------------------|-----------------|-----------------|
| portrait | 1 | **100%** | 9.0 |
| birds_flying | 2 | 86% | 9.0 |
| woman_waving | 2 | 86% | 9.0 |
| corgi_beach | 3 | 71% | 9.0 |
| landscape | 3 | 71% | 9.0 |
| mig21_missile | 3 | 71% | 7.0 |

#### Steps Agreement

| Video | Unique Optimal Steps | Agreement Score | Most Common Steps |
|-------|----------------------|-----------------|-------------------|
| birds_flying | 1 | **100%** | 15 |
| landscape | 1 | **100%** | 15 |
| corgi_beach | 2 | 86% | 15 |
| mig21_missile | 3 | 71% | 20 |
| portrait | 3 | 71% | 40 |
| woman_waving | 3 | 71% | 50 |

**Key Insight:** When metrics agree strongly (agreement score >85%), we have high confidence in the recommendation. Disagreement indicates content-specific behavior.

---

## Final Recommendations

### Content-Type Specific Settings

| Content Type | Examples | CFG | Steps | Enhanced Prompts |
|--------------|----------|-----|-------|------------------|
| **Static scenes** | portrait, landscape | 9.0 | 15-25 | ❌ Don't use |
| **Natural motion** | birds, dogs, waving | 9.0 | 15-20 | ✅ Use |
| **Fast/complex action** | jets, explosions | 7.0-7.5 | 20-30 | ❌ Don't use |

### Per-Video Recommendations (Data-Driven)

| Video | CFG | Confidence | Steps | Confidence | Use Enhanced Prompt |
|-------|-----|------------|-------|------------|---------------------|
| birds_flying | 9.0 | 88% | 15 | 100% | Yes |
| corgi_beach | 9.0 | 63% | 15 | 63% | Yes |
| landscape | 9.0 | 75% | 15 | 100% | Optional |
| mig21_missile | 7.0 | 63% | 20 | 38% | No |
| portrait | 9.0 | 100% | 40 | 63% | No |
| woman_waving | 9.0 | 88% | 50 | 38% | Yes |

---

## Key Takeaways

### 1. Higher CFG is Better (Counter-Intuitive)
- Initial hypothesis was wrong
- CFG 9.0 produces more stable, consistent frames
- Stronger guidance = less "wandering" between frames

### 2. Fewer Steps is Better (Counter-Intuitive)
- Initial hypothesis was wrong
- Steps 15-20 optimal for most content
- More steps = more opportunity for frame drift

### 3. Prompt Engineering Requires Care
- Works well for dynamic natural content
- **Backfires** on static content and fast action
- Must be tuned per content type

### 4. Multi-Metric Validation is Essential
- Single-metric analysis can mislead
- 8 metrics agreeing = high confidence
- Disagreement indicates edge cases

### 5. Content-Aware Configuration is Necessary
- No single setting works for all content
- Static vs dynamic content behave differently
- Consider automated content classification for production

---

## Files and Artifacts

### Scripts
- `05_grid_search_ablation.py` - Video generation
- `06_measure_grid_search.py` - Metrics computation
- `07_analyze_grid_search.py` - Basic analysis
- `08_analyze_comprehensive.py` - Full multi-metric analysis

### Data
- `outputs/06_grid_search_metrics/grid_search_results.json` - Raw metrics
- `outputs/08_comprehensive_analysis/*.csv` - All analysis tables

### GIFs for Presentation
Copy selected GIFs to `assets/` folder:
```bash
mkdir assets
cp outputs/05_grid_search/portrait_cfg5.0_steps25/portrait_cfg5.0_steps25.gif assets/
cp outputs/05_grid_search/portrait_cfg9.0_steps25/portrait_cfg9.0_steps25.gif assets/
# ... etc
```

---

## Future Work

1. **Automated Content Classification** - Train classifier to select optimal settings
2. **Adaptive CFG/Steps** - Vary parameters during generation based on frame content
3. **Prompt Templates** - Content-type specific prompt libraries
4. **Additional Models** - Test findings on other video diffusion models

---

*Report generated from systematic grid search ablation study.*

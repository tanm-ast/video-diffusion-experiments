# Video Diffusion Experiments

Project exploring temporal consistency in video diffusion models.

## Motivation

Investigating solutions for inter-frame consistency issues in diffusion-based video generation, specifically:
1. Understanding *why* temporal inconsistency (flickering) occurs architecturally
2. Building measurement tools to quantify inconsistency
3. Experimenting with post-processing and inference-time interventions

## Project Structure

```
.
├── docs/                    # Theory and learning notes
│   ├── 01_diffusion_fundamentals.md
│   ├── 02_video_diffusion_architecture.md
│   └── 03_temporal_consistency.md
├── experiments/             # Runnable experiment scripts
│   ├── 01_baseline_generation.py
│   ├── 02_architecture_inspection.py
│   ├── 03_measurement_pipeline.py
│   └── ...
├── outputs/                 # Generated videos and analysis (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n video_diff python=3.10 -y
conda activate video_diff
pip install -r requirements.txt
```

## Hardware

Developed on:
- NVIDIA GeForce RTX 3060 (12GB VRAM)
- 28GB system RAM
- CUDA 12.6 driver

## Learning Log

### Phase 1: Foundations
- [x] Environment setup
- [x] Baseline video generation with AnimateDiff
- [x] Architecture inspection
- [ ] Measurement pipeline

### Phase 2: Interventions
- [ ] Post-processing (flow-guided blending)
- [ ] Inference-time modifications (CFG modulation)
- [ ] Embedding analysis

## Key Findings

*To be updated as experiments progress.*

## References

- [AnimateDiff Paper](https://arxiv.org/abs/2307.04725)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)
- [Score-Based Generative Models (Song et al.)](https://arxiv.org/abs/2011.13456)

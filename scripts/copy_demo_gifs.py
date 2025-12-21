"""
Copy demonstration GIFs to assets folder for the report.

These GIFs demonstrate the key findings:
1. Higher CFG is better (CFG 5.0 vs 9.0)
2. Fewer steps is better (Steps 15 vs 50)
3. Prompt engineering is content-dependent (helps vs hurts)

Usage:
    python scripts/copy_demo_gifs.py
"""

import shutil
from pathlib import Path

# Source and destination
GRID_SEARCH_DIR = Path("outputs/05_grid_search")
ASSETS_DIR = Path("assets")

# GIFs to copy for demonstration
DEMO_GIFS = [
    # CFG comparison - Portrait (static)
    "portrait_cfg5.0_steps25/portrait_cfg5.0_steps25.gif",
    "portrait_cfg9.0_steps25/portrait_cfg9.0_steps25.gif",
    
    # CFG comparison - Birds (dynamic)
    "birds_flying_cfg5.0_steps25/birds_flying_cfg5.0_steps25.gif",
    "birds_flying_cfg9.0_steps25/birds_flying_cfg9.0_steps25.gif",
    
    # Steps comparison - Landscape (static)
    "landscape_cfg7.5_steps15/landscape_cfg7.5_steps15.gif",
    "landscape_cfg7.5_steps50/landscape_cfg7.5_steps50.gif",
    
    # Steps comparison - Birds (dynamic)
    "birds_flying_cfg7.5_steps15/birds_flying_cfg7.5_steps15.gif",
    "birds_flying_cfg7.5_steps50/birds_flying_cfg7.5_steps50.gif",
    
    # Prompt comparison - Woman waving (HELPS)
    "woman_waving_cfg7.5_steps25_prompt_baseline/woman_waving_cfg7.5_steps25_prompt_baseline.gif",
    "woman_waving_cfg7.5_steps25_prompt_enhanced/woman_waving_cfg7.5_steps25_prompt_enhanced.gif",
    
    # Prompt comparison - Portrait (HURTS)
    "portrait_cfg7.5_steps25_prompt_baseline/portrait_cfg7.5_steps25_prompt_baseline.gif",
    "portrait_cfg7.5_steps25_prompt_enhanced/portrait_cfg7.5_steps25_prompt_enhanced.gif",
    
    # Prompt comparison - MiG-21 (HURTS)
    "mig21_missile_cfg7.5_steps25_prompt_baseline/mig21_missile_cfg7.5_steps25_prompt_baseline.gif",
    "mig21_missile_cfg7.5_steps25_prompt_enhanced/mig21_missile_cfg7.5_steps25_prompt_enhanced.gif",
    
    # Prompt comparison - Corgi (HELPS)
    "corgi_beach_cfg7.5_steps25_prompt_baseline/corgi_beach_cfg7.5_steps25_prompt_baseline.gif",
    "corgi_beach_cfg7.5_steps25_prompt_enhanced/corgi_beach_cfg7.5_steps25_prompt_enhanced.gif",
]


def main():
    # Create assets directory
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying demonstration GIFs to {ASSETS_DIR}/")
    print("=" * 60)
    
    copied = 0
    missing = 0
    
    for gif_path in DEMO_GIFS:
        src = GRID_SEARCH_DIR / gif_path
        # Flatten the filename (remove subdirectory)
        dst_name = gif_path.split("/")[-1]
        dst = ASSETS_DIR / dst_name
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied: {dst_name}")
            copied += 1
        else:
            print(f"✗ Missing: {src}")
            missing += 1
    
    print("=" * 60)
    print(f"Copied: {copied}, Missing: {missing}")
    
    if missing > 0:
        print("\nNote: Missing GIFs may not have been generated yet.")
        print("Run experiments/05_grid_search_ablation.py first.")
    
    # Print organization for report
    print("\n" + "=" * 60)
    print("GIF ORGANIZATION FOR REPORT")
    print("=" * 60)
    
    print("\n## CFG Comparison (Higher is Better)")
    print("| Portrait CFG 5.0 | Portrait CFG 9.0 |")
    print("|------------------|------------------|")
    print("| portrait_cfg5.0_steps25.gif | portrait_cfg9.0_steps25.gif |")
    
    print("\n## Steps Comparison (Fewer is Better)")
    print("| Landscape Steps 15 | Landscape Steps 50 |")
    print("|--------------------|---------------------|")
    print("| landscape_cfg7.5_steps15.gif | landscape_cfg7.5_steps50.gif |")
    
    print("\n## Prompt Engineering (Content-Dependent)")
    print("| Content | Baseline | Enhanced | Effect |")
    print("|---------|----------|----------|--------|")
    print("| woman_waving | _prompt_baseline.gif | _prompt_enhanced.gif | HELPS |")
    print("| portrait | _prompt_baseline.gif | _prompt_enhanced.gif | HURTS |")
    print("| mig21_missile | _prompt_baseline.gif | _prompt_enhanced.gif | HURTS |")


if __name__ == "__main__":
    main()

# Demo Outputs

This directory contains outputs from running the demo notebook (`notebooks/04_live_demo.ipynb`) on Google Colab.

## Directory Structure

```
demo_outputs/
├── videos/                                        # MP4 recordings of trained agents
│   ├── halfcheetah_2Q_baseline.mp4                #   HalfCheetah — normal physics
│   ├── halfcheetah_2Q_friction0.5.mp4             #   HalfCheetah — 0.5× friction
│   ├── halfcheetah_2Q_gravity2x.mp4               #   HalfCheetah — 2× gravity
│   ├── halfcheetah_2Q_noise0.3.mp4                #   HalfCheetah — obs noise σ=0.3
│   ├── hopper_2Q_baseline.mp4                     #   Hopper — normal physics
│   ├── hopper_2Q_friction0.5.mp4                  #   Hopper — 0.5× friction
│   ├── hopper_2Q_gravity2x.mp4                    #   Hopper — 2× gravity
│   ├── hopper_2Q_noise0.3.mp4                     #   Hopper — obs noise σ=0.3
│   ├── walker2d_2Q_baseline.mp4                   #   Walker2d — normal physics
│   ├── walker2d_2Q_friction0.5.mp4                #   Walker2d — 0.5× friction
│   ├── walker2d_2Q_gravity2x.mp4                  #   Walker2d — 2× gravity
│   └── walker2d_2Q_noise0.3.mp4                   #   Walker2d — obs noise σ=0.3
│
├── plots/                                         # PNG charts from demo analysis
│   ├── training_curves_halfcheetah.png             #   2Q vs 3Q training progress
│   ├── training_curves_hopper.png
│   ├── training_curves_walker2d.png
│   ├── degradation_curves_halfcheetah.png          #   Performance vs shift severity
│   ├── degradation_curves_hopper.png
│   ├── degradation_curves_walker2d.png
│   ├── robustness_drop_bars_halfcheetah.png        #   AUDC bar charts (2Q vs 3Q)
│   ├── robustness_drop_bars_hopper.png
│   ├── robustness_drop_bars_walker2d.png
│   ├── demo_vs_production_halfcheetah.png          #   Demo vs full-run comparison
│   ├── demo_vs_production_hopper.png
│   └── demo_vs_production_walker2d.png
│
└── csvs/                                          # Shift evaluation results
    ├── demo_shift_halfcheetah-medium-v2_2Q.csv     #   2Q shift eval (all 4 shifts)
    ├── demo_shift_halfcheetah-medium-v2_3Q.csv     #   3Q shift eval
    ├── demo_shift_hopper-medium-v2_2Q.csv
    ├── demo_shift_hopper-medium-v2_3Q.csv
    ├── demo_shift_walker2d-medium-v2_2Q.csv
    ├── demo_shift_walker2d-medium-v2_3Q.csv
    ├── demo_training_halfcheetah-medium-v2_2Q.csv  #   2Q training metrics
    ├── demo_training_halfcheetah-medium-v2_3Q.csv  #   3Q training metrics
    ├── demo_training_hopper-medium-v2_2Q.csv
    ├── demo_training_hopper-medium-v2_3Q.csv
    ├── demo_training_walker2d-medium-v2_2Q.csv
    └── demo_training_walker2d-medium-v2_3Q.csv
```

## File Summary

| Category | Count | Total Size | Description |
|----------|-------|------------|-------------|
| Videos   | 12    | ~5 MB      | MP4 recordings (4 conditions × 3 environments) |
| Plots    | 12    | ~2.3 MB    | PNG charts (4 plot types × 3 environments) |
| CSVs     | 12    | ~8 KB      | Shift eval + training metrics (2 types × 2 configs × 3 envs) |

## How to Generate

1. Open `notebooks/04_live_demo.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run All cells (~2-3 hours)
4. Download `demo_outputs.zip` from the last cell
5. Unzip into this directory

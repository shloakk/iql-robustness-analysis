# Demo Outputs

This directory contains outputs from running the demo notebook (`notebooks/04_live_demo.ipynb`) on Google Colab.

## Directory Structure

```
demo_outputs/
├── plots/          # PNG charts (training curves, degradation, robustness)
│   ├── training_curves_hopper-medium-v2.png
│   ├── training_curves_halfcheetah-medium-v2.png
│   ├── training_curves_walker2d-medium-v2.png
│   ├── degradation_curves_hopper-medium-v2.png
│   ├── degradation_curves_halfcheetah-medium-v2.png
│   ├── degradation_curves_walker2d-medium-v2.png
│   ├── robustness_drop_bars_hopper-medium-v2.png
│   ├── robustness_drop_bars_halfcheetah-medium-v2.png
│   └── robustness_drop_bars_walker2d-medium-v2.png
├── videos/         # MP4 recordings of trained agents
│   ├── hopper-medium-v2_2Q_baseline.mp4
│   ├── hopper-medium-v2_2Q_gravity.mp4
│   ├── hopper-medium-v2_2Q_noise.mp4
│   ├── hopper-medium-v2_2Q_friction.mp4
│   ├── halfcheetah-medium-v2_2Q_baseline.mp4
│   ├── ... (4 videos per environment)
│   └── walker2d-medium-v2_2Q_friction.mp4
└── csvs/           # Shift evaluation results
    ├── demo_shift_hopper-medium-v2_2Q.csv
    ├── demo_shift_hopper-medium-v2_3Q.csv
    ├── ... (2 CSVs per environment)
    └── demo_training_walker2d-medium-v2_3Q.csv
```

## How to Generate

1. Open `notebooks/04_live_demo.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run All cells (~2-3 hours)
4. Download `demo_outputs.zip` from the last cell
5. Unzip into this directory

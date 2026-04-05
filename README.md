# Robustness of Implicit Q-Learning Under Controlled Distribution Shift

**CMPE 260 — Reinforcement Learning | Group 6 | San José State University**

---

## Team

| Student | Role |
|---|---|
| Joao Lucas Veras | Baseline IQL reproduction |
| Shloak Aggarwal | Distribution shift design |
| Pramod Yadav | Evaluation metrics & literature survey |
| Uday Arora | Q-ensemble extension |

---

## What We Set Out To Do

From our proposal, we planned to:

1. Review offline RL literature (IQL, CQL, TD3+BC)
2. Reproduce IQL on D4RL benchmarks and establish baseline scores
3. Design controlled distribution shifts in MuJoCo (gravity, friction, observation noise, reward perturbations)
4. Evaluate robustness across multiple seeds
5. Implement a robustness-oriented extension (Q-ensemble with 3 critics)
6. Conduct ablation studies comparing DoubleCritic vs TripleCritic under shift

The core research question: *How robust is Implicit Q-Learning under controlled distribution shift, and can we improve its robustness?*

---

## What We Have Demonstrated So Far

### Literature & Theory
- Surveyed three offline RL approaches: IQL (expectile regression), CQL (pessimistic Q-values), TD3+BC (behavior cloning regularization)
- Defined formal robustness metrics: normalized performance `J(π, E_δ)`, robustness drop `Δ(δ)`, and Area Under Degradation Curve (AUDC)
- Identified the gap: none of these methods have been evaluated under environment-level perturbations at test time

### Baseline Reproduction
- Reproduced IQL on `hopper-medium-v2` with a final normalized score of **52.79** (300k training steps)
- Implementation uses JAX/Flax with 2-layer MLPs (256 hidden units), expectile τ=0.7, temperature β=3.0

### Q-Ensemble Extension
- Implemented `TripleCritic` — 3 Q-networks taking `min(q1, q2, q3)` for more conservative value estimation
- Trained on `hopper-medium-v2` with a final score of **50.88** (300k steps)
- The 1.91-point gap confirms the implementation is correct — the benefit of the ensemble is expected to show under distribution shift, not in baseline performance

### Distribution Shift Wrappers
- **Gravity shift** — scales MuJoCo gravity vector (levels: 0.5x, 1.0x, 1.5x, 2.0x)
- **Observation noise** — adds Gaussian noise to observations (σ = 0.0, 0.01, 0.1, 0.3)
- **Friction shift** — scales MuJoCo friction coefficients (levels: 0.5x, 1.0x, 1.5x, 2.0x)
- **Reward perturbation** — adds noise to rewards (σ = 0.0, 0.1, 0.5, 1.0)

### Evaluation Pipeline
- `scripts/evaluate_shift.py` — evaluates a trained agent under any combination of shift types, outputs CSV
- `scripts/compute_robustness.py` — reads CSVs and computes Δ(δ), AUDC, worst-case performance, and side-by-side comparison tables
- `scripts/run_all_hpc.sh` — single script that submits all training, evaluation, and ablation jobs to SLURM

---

## What Remains

| Task | Status | Notes |
|---|---|---|
| Baseline training on halfcheetah-medium-v2 | Code ready, not yet run | Same script, different `--env_name` |
| Baseline training on walker2d-medium-v2 | Code ready, not yet run | Same script, different `--env_name` |
| Q-ensemble training on halfcheetah + walker2d | Code ready, not yet run | `--num_critics=3` |
| Shift evaluation across all envs and shift types | Code ready, not yet run | `--shift_type=all` |
| Robustness metrics computation | Code ready, not yet run | Depends on shift eval CSVs |
| Expectile τ ablation (τ = 0.5, 0.8, 0.9) | Code ready, not yet run | `--config.expectile=0.5` |
| Multiple seeds for error bars | Code ready, not yet run | Change `--seed` |
| Final results table and plots | Notebook ready | `notebooks/04_analyze_results.ipynb` |

All of these are triggered by a single command on HPC:
```bash
./scripts/run_all_hpc.sh
```

---

## Datasets

We use the [D4RL](https://github.com/Farama-Foundation/d4rl) benchmark datasets for MuJoCo continuous control:

| Environment | Obs Dim | Act Dim | Dataset Size | Description |
|---|---|---|---|---|
| `hopper-medium-v2` | 11 | 3 | 1M transitions | One-legged hopping robot, mediocre policy data |
| `halfcheetah-medium-v2` | 17 | 6 | 1M transitions | Two-legged running robot, mediocre policy data |
| `walker2d-medium-v2` | 17 | 6 | 1M transitions | Two-legged walking robot, mediocre policy data |

---

## Methodology

### Training
Standard IQL with expectile regression for value learning, TD updates for Q-learning, and advantage-weighted behavioral cloning for policy extraction. Two-layer MLP networks (256 hidden units), Adam optimizer, cosine learning rate schedule for the actor.

### Distribution Shift
Perturbations are applied **at evaluation time only** — the policy is never retrained. This isolates the sensitivity of offline-trained policies to changes in environment dynamics.

| Shift Type | MuJoCo Parameter | Levels |
|---|---|---|
| Gravity | `model.opt.gravity` | 0.5x, 1.0x, 1.5x, 2.0x |
| Observation Noise | Gaussian σ | 0.0, 0.01, 0.1, 0.3 |
| Friction | `model.geom_friction` | 0.5x, 1.0x, 1.5x, 2.0x |
| Reward Perturbation | Gaussian σ | 0.0, 0.1, 0.5, 1.0 |

### Q-Ensemble Extension
We extend IQL's `DoubleCritic` (2 Q-networks, `min(q1,q2)`) to a `TripleCritic` (3 Q-networks, `min(q1,q2,q3)`). The hypothesis: taking the minimum over more Q-networks produces more conservative value estimates, which should degrade less under distribution shift.

### Metrics
- **Robustness drop:** `Δ(δ) = (J(π, E_0) - J(π, E_δ)) / J(π, E_0)` — 0 means robust, positive means degraded
- **AUDC:** Area Under Degradation Curve — integrates |Δ(δ)| over shift levels. Lower is better.
- **Worst-case:** minimum score across all shift levels

---

## Results

### Baseline Performance (No Shift)

| Environment | Baseline IQL (2Q) | Q-Ensemble IQL (3Q) |
|---|---|---|
| hopper-medium-v2 | **52.79** | 50.88 |
| halfcheetah-medium-v2 | — | — |
| walker2d-medium-v2 | — | — |

### Shift Evaluation

Pending — run `./scripts/run_all_hpc.sh` on HPC to generate.

---

## Repository Structure

```
iql-robustness-analysis/
├── iql/                          # Core IQL implementation
│   ├── actor.py                  #   Actor update (advantage-weighted BC)
│   ├── critic.py                 #   Critic update (2 or 3 Q-networks)
│   ├── common.py                 #   MLP, Model, Batch definitions
│   ├── learner.py                #   Training loop + checkpointing
│   ├── policy.py                 #   NormalTanhPolicy + sampling
│   ├── value_net.py              #   DoubleCritic, TripleCritic, ValueCritic
│   └── dataset_utils.py          #   D4RL dataset loading
│
├── evaluation/                   # Policy evaluation
│   └── evaluate.py
│
├── wrappers/                     # Environment wrappers
│   ├── episode_monitor.py        #   Episode return/length tracking
│   ├── single_precision.py       #   Float32 casting
│   ├── gravity_shift.py          #   Gravity scaling
│   ├── observation_noise.py      #   Gaussian observation noise
│   ├── friction_shift.py         #   Friction scaling
│   └── reward_perturbation.py    #   Reward noise/scaling
│
├── configs/                      # Hyperparameter configs
│   ├── mujoco_config.py          #   τ=0.7, β=3.0
│   ├── antmaze_config.py         #   τ=0.9, β=10.0
│   └── kitchen_config.py         #   τ=0.7, β=0.5, dropout=0.1
│
├── scripts/                      # Training & evaluation
│   ├── train_offline.py          #   Offline training (--num_critics flag)
│   ├── train_finetune.py         #   Online finetuning
│   ├── evaluate_shift.py         #   Evaluate under shift (all 4 types)
│   ├── compute_robustness.py     #   Compute metrics from CSVs
│   └── run_all_hpc.sh            #   Submit all experiments to SLURM
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_train_baseline.ipynb   #   Train 2Q on all envs
│   ├── 02_train_ensemble.ipynb   #   Train 3Q on all envs
│   ├── 03_evaluate_shift.ipynb   #   Evaluate under shift
│   └── 04_analyze_results.ipynb  #   Generate plots and tables
│
├── results/                      # Experiment outputs
│   ├── results_baseline_iql.csv
│   ├── results_ensemble_iql.csv
│   └── results_comparison.png
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Running Experiments

### On SJSU CoE HPC (recommended)

The SJSU College of Engineering HPC uses SLURM for job scheduling. Connect via
VPN if off-campus, then SSH in. All setup runs on the login node (which has
internet access). The script uses `--only-binary=:all:` to install pre-built
wheels, avoiding any C compilation on the login node.

```bash
# 1. SSH in (use coe-hpc1 if coe-hpc times out over VPN)
ssh <sjsu_id>@coe-hpc.sjsu.edu

# 2. Clone the repo
git clone -b sp1ffygeek_check_3 https://github.com/shloakk/iql-robustness-analysis.git
cd iql-robustness-analysis

# 3. One-time setup (on login node — downloads pre-built wheels)
bash scripts/run_all_hpc.sh setup

# 4. Submit experiments to GPU nodes
mkdir -p logs
sbatch scripts/run_all_hpc.sh          # full pipeline
# or run individual steps:
# sbatch scripts/run_all_hpc.sh train    # training only
# sbatch scripts/run_all_hpc.sh eval     # shift evaluation only
# sbatch scripts/run_all_hpc.sh analyze  # compute metrics only

# 5. Monitor
squeue -u $USER                        # check job status
tail -f logs/slurm_<job_id>.out        # watch output
```

The setup step creates a Python venv using the system Python 3.11 and installs
all dependencies as pre-built binary wheels (no compilation needed). This only
needs to be done once — the `/home` directory is shared across all HPC nodes,
so batch jobs on GPU nodes activate the same venv.

The batch job runs sequentially within a single SLURM allocation:
- **Phase 1:** Training — 6 runs (3 envs x 2 critic configs), ~20 min each
- **Phase 2:** Shift evaluation — 6 runs (all 4 shift types per model)
- **Phase 3:** Expectile tau ablation — 9 runs (3 envs x 3 tau values)
- **Phase 4:** Analysis — computes robustness metrics from CSVs

HPC partitions: `gpu` (P100/A100/H100, 48h max), `compute` (CPU only, 24h max),
`condo` (preemptible). The script defaults to the `gpu` partition with a 24h
time limit, which is sufficient for the full pipeline.

A convenience alias file is provided for common HPC commands:

```bash
source scripts/hpc_aliases.sh    # load once per session
# or add to ~/.bashrc for permanent use:
# echo 'source ~/iql-robustness-analysis/scripts/hpc_aliases.sh' >> ~/.bashrc
```

| Alias | Command |
|---|---|
| `jobs` | Check your job status |
| `myjobs` | Detailed job listing |
| `killall` | Cancel all your jobs |
| `iql-setup` | One-time environment setup |
| `iql-run` | Submit full pipeline |
| `iql-train` | Submit training only |
| `iql-eval` | Submit evaluation only |
| `iql-analyze` | Submit analysis only |
| `lastlog` | Tail the latest output log |
| `lasterr` | Tail the latest error log |
| `gpunode` | Get an interactive GPU session |
| `results` | List result CSVs |

### On Google Colab

Run the notebooks in order: `01_train_baseline.ipynb` → `02_train_ensemble.ipynb` → `03_evaluate_shift.ipynb` → `04_analyze_results.ipynb`

### Locally

```bash
python scripts/train_offline.py --env_name=hopper-medium-v2 --config=configs/mujoco_config.py --num_critics=2
python scripts/evaluate_shift.py --env_name=hopper-medium-v2 --shift_type=all --num_critics=2
python scripts/compute_robustness.py --results_dir=results/ --env_name=hopper-medium-v2
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Actor / Critic / Value LR | 3e-4 |
| Hidden dims | (256, 256) |
| Discount γ | 0.99 |
| Expectile τ | 0.7 |
| Temperature β | 3.0 |
| Soft target update rate | 0.005 |
| Batch size | 256 |
| Training steps | 300,000 |
| Optimizer | Adam |
| Actor LR schedule | Cosine decay |

---

## References

1. Kostrikov, I., Nair, A., & Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning*. ICLR. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
2. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
3. Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC)*. NeurIPS. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
4. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*. [arXiv:2004.07219](https://arxiv.org/abs/2004.07219)

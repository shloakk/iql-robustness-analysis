# Slide-by-Slide Speaker Notes

**Presentation:** *Robustness of Implicit Q-Learning Under Controlled Distribution Shift*
**Course:** CMPE 260 — Reinforcement Learning | San José State University | Group 6
**Duration:** 5–7 minutes
**Companion slides:** [`slides/Slides_Robustness-of-Implicit-Q-Learning-Under-Controlled-Distribution-Shift.pptx`](../slides/Slides_Robustness-of-Implicit-Q-Learning-Under-Controlled-Distribution-Shift.pptx)

---

## Slide 1: Title & Motivation

**⏱ Time: 30–45 seconds**

### What to say

- "Our project studies the **robustness of Implicit Q-Learning** — a popular offline RL algorithm — when the deployment environment differs from training."
- "In offline RL, we train from a fixed dataset and never interact with the environment during training. But what happens when the real world doesn't match the training data?"
- "This is the **distribution shift** problem — and it's the central challenge for deploying offline RL in practice."
- "Think of a robot that learned to walk on a smooth floor. What happens when you put it on ice? Or on a different planet with stronger gravity?"
- "We systematically measure this degradation and test whether a simple modification — adding a third Q-network — can improve robustness."

### Key facts to mention

- Offline RL learns from fixed datasets — no environment interaction during training
- Distribution shift = mismatch between training and deployment conditions
- This is a **practical** problem: sim-to-real transfer, sensor degradation, changing environments

### Transition

> "Let me first explain how IQL works and why it's a good candidate for studying robustness."

---

## Slide 2: IQL Background

**⏱ Time: 45–60 seconds**

### What to say

- "IQL has three key components. First, it learns a **value function V(s)** using expectile regression."
- *(Point to equation)* "This asymmetric loss `L_τ(u) = |τ - 𝟙(u<0)| · u²` biases V toward higher-return actions. The hyperparameter τ controls how optimistic the estimate is — τ=0.5 is the median, τ=0.9 targets the 90th percentile."
- "Second, it learns **Q-functions** using standard TD updates: `Q(s,a) ← r + γV(s')`."
- "Third, it extracts a **policy** via advantage-weighted behavioral cloning: actions where Q > V get exponentially higher weight, controlled by temperature β."
- "The key innovation: IQL **never queries out-of-distribution actions** during training. Unlike CQL which explicitly penalizes OOD actions, or TD3+BC which constrains the policy, IQL avoids the problem entirely by keeping value learning within the dataset support."

### Key equations to reference

| Component | Equation |
|---|---|
| Value function | `V(s) = argmin_v E[L_τ(Q(s,a) - v)]` |
| Expectile loss | `L_τ(u) = \|τ - 𝟙(u<0)\| · u²` |
| Q-update | `Q(s,a) ← r(s,a) + γV(s')` |
| Policy | `max_π E[exp(β(Q(s,a) - V(s))) · log π(a\|s)]` |

### Key facts to mention

- Default hyperparameters: τ = 0.7, β = 3.0, hidden dims = (256, 256)
- Implementation uses JAX/Flax with Adam optimizer and cosine LR schedule for the actor
- Standard DoubleCritic: 2 Q-networks, take min(Q₁, Q₂)

### Transition

> "Our main extension is simple but effective — we add a third Q-network."

---

## Slide 3: Our Extension — TripleCritic

**⏱ Time: 45–60 seconds**

### What to say

- "Standard IQL uses a **DoubleCritic** — two Q-networks, taking the minimum. This is a well-known technique from TD3 to reduce overestimation."
- "We extend this to a **TripleCritic** — three Q-networks, taking `min(Q₁, Q₂, Q₃)`."
- "The intuition: the minimum of 3 independent estimates is stochastically smaller than the minimum of 2. This produces a **more conservative** value estimate."
- "Why does conservatism help under distribution shift? When the agent encounters states it hasn't seen in training, an optimistic Q-function might overestimate the value of actions — leading to catastrophic failures. A more conservative estimate provides a safety margin."
- "The implementation is straightforward — in [`iql/value_net.py`](../iql/value_net.py), we define a [`TripleCritic`](../iql/value_net.py:45) class that instantiates 3 independent [`Critic`](../iql/value_net.py:18) networks. In [`iql/critic.py`](../iql/critic.py), the [`_min_q()`](../iql/critic.py:13) function handles both 2 and 3 critics transparently."
- "The [`Learner`](../iql/learner.py:53) class takes a `num_critics` parameter — set to 2 for DoubleCritic, 3 for TripleCritic."

### Key facts to mention

- 50% more critic parameters, ~15-20% more total training time
- Hypothesis: more conservative Q → less overestimation → better robustness under shift
- The extension is a **drop-in replacement** — no changes to the training algorithm

### Transition

> "Now let's look at how we test robustness — our distribution shift protocol."

---

## Slide 4: Distribution Shift Protocol

**⏱ Time: 45–60 seconds**

### What to say

- "We apply **four types of perturbation at test time only** — the policy is never retrained. This isolates the sensitivity of offline-trained policies to environment changes."
- *(Point to each shift type)*
  - "**Gravity shift**: scale the MuJoCo gravity vector — 0.5× (moon-like) to 2.0× (Jupiter-like). Tests robustness to fundamental physics changes."
  - "**Friction shift**: scale all friction coefficients — 0.5× (icy floor) to 2.0× (sandpaper). Tests contact dynamics robustness."
  - "**Observation noise**: add Gaussian noise σ ∈ {0, 0.01, 0.1, 0.3} to every sensor reading. Tests perceptual robustness."
  - "**Reward perturbation**: add noise to rewards. This is a **control experiment** — since the policy is frozen, reward noise should have zero effect. And it does: AUDC < 0.002 everywhere."
- "These wrappers are implemented in [`wrappers/gravity_shift.py`](../wrappers/gravity_shift.py) and [`wrappers/observation_noise.py`](../wrappers/observation_noise.py) — clean gym wrappers that modify the environment at reset time."
- "We measure robustness using **AUDC** — Area Under Degradation Curve. It integrates the normalized performance drop across all shift levels. Lower AUDC = more robust."

### Key facts to mention

- 4 shift types × 4 severity levels = 16 evaluation conditions per model
- AUDC formula: integrate `|Δ(δ)| = |(J(E₀) - J(E_δ)) / J(E₀)|` over shift levels
- Reward perturbation AUDC < 0.002 confirms the policy is truly offline

### Transition

> "Let me quickly walk through the codebase architecture before we dive into results."

---

## Slide 5: Codebase Architecture

**⏱ Time: 45–60 seconds**

### What to say

- "The project is organized into five main directories."
- *(Point to each)*
  - "**[`iql/`](../iql/)** — the core IQL implementation in JAX/Flax. [`learner.py`](../iql/learner.py) orchestrates training, [`value_net.py`](../iql/value_net.py) defines DoubleCritic and TripleCritic, [`critic.py`](../iql/critic.py) handles Q and V updates with the expectile loss, and [`actor.py`](../iql/actor.py) implements advantage-weighted policy extraction."
  - "**[`wrappers/`](../wrappers/)** — gym wrappers for each shift type. [`GravityShift`](../wrappers/gravity_shift.py:15) modifies `model.opt.gravity` at reset; [`ObservationNoise`](../wrappers/observation_noise.py:15) adds Gaussian noise to observations."
  - "**[`evaluation/`](../evaluation/)** — the [`evaluate.py`](../evaluation/evaluate.py) module runs episodes and collects returns."
  - "**[`scripts/`](../scripts/)** — the pipeline scripts. [`train_offline.py`](../scripts/train_offline.py) trains models, [`evaluate_shift.py`](../scripts/evaluate_shift.py) runs shift evaluations, [`compute_robustness.py`](../scripts/compute_robustness.py) computes AUDC metrics, and [`run_all_hpc.sh`](../scripts/run_all_hpc.sh) orchestrates everything on SLURM."
  - "**[`notebooks/`](../notebooks/)** — Jupyter notebooks for interactive training, evaluation, analysis, and demo."
- "The pipeline runs in 4 phases: **Train → Evaluate under shift → τ ablation → Compute metrics**. A single `run_all_hpc.sh` command runs everything."

### Key facts to mention

- All training: 300k steps on SJSU CoE HPC GPU partition
- 4 seeds (42, 43, 44, 45) for statistical reliability
- Baseline measurement is embedded in shift evaluation (gravity=1.0× is one of the 4 levels)
- Total: **1,536 shift-level evaluations**, each averaged over 10 episodes

### Transition

> "Now for the results — what did we find?"

---

## Slide 6: Results

**⏱ Time: 60–90 seconds**

### What to say

- "We tested 3 environments × 8 configurations × 4 shift types × 4 levels × 4 seeds = **1,536 evaluations**."

#### Baseline Performance

- "First, baseline performance without any shift. 2Q and 3Q achieve comparable returns:"
  - "Hopper: 2Q **1571 ± 136**, 3Q **1469 ± 38** — note the much tighter std for 3Q"
  - "HalfCheetah: 2Q **5543 ± 31**, 3Q **5501 ± 35** — nearly identical"
  - "Walker2d: 2Q **3360 ± 152**, 3Q **3423 ± 104** — 3Q slightly higher"

#### 2Q vs 3Q Robustness

- "The key finding: **3Q consistently improves robustness on Hopper**."
  - "Gravity AUDC: 3Q **0.529 ± 0.069** vs 2Q **0.616 ± 0.027** — a 14% improvement"
  - "Friction AUDC: 3Q **0.692 ± 0.011** vs 2Q **0.713 ± 0.026** — tighter error bars too"
- "But this is **environment-dependent**:"
  - "On HalfCheetah, 2Q and 3Q are statistically indistinguishable — all differences within error bars"
  - "On Walker2d, 2Q is actually more robust on gravity (0.716 vs 0.739) and friction (0.109 vs 0.131)"

#### τ Ablation

- "The expectile ablation reveals a clear trade-off:"
  - "Lower τ → lower baseline but better robustness. Hopper 2Q: τ=0.5 gravity AUDC **0.538** vs τ=0.8 **0.730**"
  - "Higher τ → higher variance. τ=0.9 gravity std = **0.097** vs τ=0.7 std = **0.027**"
- "The best overall configuration for Hopper is **3Q at τ=0.7**: gravity AUDC 0.529, friction AUDC 0.692 with the tightest error bars"
- "But **no single configuration dominates** across all environments and shift types"

### Key numbers to have ready

| Metric | Value |
|---|---|
| Total evaluations | 1,536 |
| Hopper 3Q gravity AUDC | 0.529 ± 0.069 |
| Hopper 2Q gravity AUDC | 0.616 ± 0.027 |
| 3Q improvement on Hopper gravity | 14% |
| HalfCheetah friction AUDC (all configs) | < 0.033 |
| Reward perturbation AUDC (all configs) | < 0.002 |
| Hopper 3Q baseline std | 38 (vs 2Q std = 136) |

### Transition

> "Let me show you what this looks like in practice with some demo videos."

---

## Slide 7: Demo Videos & Conclusion

**⏱ Time: 30–45 seconds**

### What to say

- *(If showing videos)* "Here's the Hopper agent under baseline conditions — smooth, stable hopping. Now watch what happens under 2× gravity — the robot can barely get off the ground. And under 0.5× friction — it slips and falls almost immediately."
- "These videos are from the 2Q agent. The 3Q agent shows similar qualitative behavior but maintains slightly better control under the same shift conditions."

#### Key Takeaways (enumerate clearly)

- "**Finding 1:** Q-ensemble (3Q) improves robustness on Hopper — confirmed across 4 seeds — but the benefit is environment-dependent."
- "**Finding 2:** Gravity and friction are the most damaging shifts (AUDC > 0.5), while reward perturbation has zero effect — confirming the policy is truly offline."
- "**Finding 3:** Lower expectile τ trades baseline performance for robustness. Higher τ increases cross-seed variance."
- "**Finding 4:** No single (critics, τ) configuration dominates — the optimal choice is environment-dependent."

#### Future Work

- "Three directions for future work:"
  - "**Compositional shifts** — testing multiple perturbations simultaneously"
  - "**Adaptive τ selection** — choosing τ based on expected deployment conditions"
  - "**Larger ensembles and other offline RL methods** — comparing CQL and TD3+BC under the same shift protocol"

### Demo videos available

| Video | Path |
|---|---|
| Hopper baseline | [`demo_outputs/videos/hopper_2Q_baseline.mp4`](../demo_outputs/videos/hopper_2Q_baseline.mp4) |
| Hopper 2× gravity | [`demo_outputs/videos/hopper_2Q_gravity2x.mp4`](../demo_outputs/videos/hopper_2Q_gravity2x.mp4) |
| Hopper 0.5× friction | [`demo_outputs/videos/hopper_2Q_friction0.5.mp4`](../demo_outputs/videos/hopper_2Q_friction0.5.mp4) |
| Hopper noise σ=0.3 | [`demo_outputs/videos/hopper_2Q_noise0.3.mp4`](../demo_outputs/videos/hopper_2Q_noise0.3.mp4) |
| HalfCheetah baseline | [`demo_outputs/videos/halfcheetah_2Q_baseline.mp4`](../demo_outputs/videos/halfcheetah_2Q_baseline.mp4) |
| Walker2d baseline | [`demo_outputs/videos/walker2d_2Q_baseline.mp4`](../demo_outputs/videos/walker2d_2Q_baseline.mp4) |

### Closing line

> "The main takeaway: robustness in offline RL is not free — it requires deliberate architectural and hyperparameter choices, and the optimal configuration depends on the deployment scenario. Thank you."

---

## Timing Summary

| Slide | Topic | Target Time | Cumulative |
|---|---|---|---|
| 1 | Title & Motivation | 30–45 sec | 0:45 |
| 2 | IQL Background | 45–60 sec | 1:45 |
| 3 | TripleCritic Extension | 45–60 sec | 2:45 |
| 4 | Distribution Shift Protocol | 45–60 sec | 3:45 |
| 5 | Codebase Architecture | 45–60 sec | 4:45 |
| 6 | Results | 60–90 sec | 6:15 |
| 7 | Demo & Conclusion | 30–45 sec | 7:00 |

**Total: 5:15 – 7:00 minutes** (target: 5–7 min)

---

## Backup Slides / Q&A Prep

If asked about specific numbers during Q&A, refer to:

- [`results/summary_hopper-medium-v2.csv`](../results/summary_hopper-medium-v2.csv) — all Hopper AUDC values
- [`results/summary_halfcheetah-medium-v2.csv`](../results/summary_halfcheetah-medium-v2.csv) — all HalfCheetah AUDC values
- [`results/summary_walker2d-medium-v2.csv`](../results/summary_walker2d-medium-v2.csv) — all Walker2d AUDC values
- [`docs/DETAILED_RESULTS.md`](DETAILED_RESULTS.md) — per-seed breakdown
- [`docs/GRAD_QUESTIONS.md`](GRAD_QUESTIONS.md) — 30 anticipated questions with answers

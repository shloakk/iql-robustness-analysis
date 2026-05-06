#!/usr/bin/env python3
"""Validate the 04_live_demo.ipynb notebook end-to-end with minimal data.

This script tests each key component of the demo notebook with very few
training steps (10) and eval episodes (2) to verify correctness without
requiring a full training run.

Expected runtime: < 5 minutes (mostly dataset download on first run).
"""

import os
import sys
import time
import traceback

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Suppress D4RL import warnings
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

RESULTS = {}
TOTAL_START = time.time()


def run_test(name, fn):
    """Run a test function and record pass/fail."""
    print(f"\n{'='*60}")
    print(f"TEST {len(RESULTS)+1}: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        RESULTS[name] = ('PASS', elapsed, None)
        print(f"  ✅ PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        RESULTS[name] = ('FAIL', elapsed, tb)
        print(f"  ❌ FAIL ({elapsed:.1f}s): {e}")
        print(tb)


# ── Test 1: Imports ──────────────────────────────────────────────────────────

def test_imports():
    """Verify all required modules can be imported."""
    import jax
    print(f"  JAX version: {jax.__version__}")
    print(f"  JAX devices: {jax.devices()}")

    import flax
    print(f"  Flax version: {flax.__version__}")

    import gymnasium as gym
    print(f"  Gymnasium version: {gym.__version__}")

    import mujoco
    print(f"  MuJoCo version: {mujoco.__version__}")

    import numpy as np
    print(f"  NumPy version: {np.__version__}")

    import optax
    print(f"  Optax version: {optax.__version__}")

    # Project-specific imports (matching notebook Cell 3)
    import wrappers
    from iql import Learner
    from iql.dataset_utils import D4RLDataset, split_into_trajectories, d4rl_to_gymnasium_name
    from evaluation import evaluate
    from wrappers.gravity_shift import GravityShift
    from wrappers.observation_noise import ObservationNoise
    from wrappers.friction_shift import FrictionShift
    from wrappers.reward_perturbation import RewardPerturbation

    print("  All project imports successful")


run_test("Imports", test_imports)


# ── Test 2: Dataset loading ──────────────────────────────────────────────────

dataset = None  # will be set by test

def test_dataset_loading():
    global dataset
    from iql.dataset_utils import D4RLDataset
    dataset = D4RLDataset('hopper-medium-v2')
    print(f"  Dataset size: {dataset.size:,} transitions")
    print(f"  Observation shape: {dataset.observations.shape}")
    print(f"  Action shape: {dataset.actions.shape}")
    assert dataset.size > 0, "Dataset is empty"
    assert dataset.observations.ndim == 2, f"Expected 2D observations, got {dataset.observations.ndim}D"
    assert dataset.actions.ndim == 2, f"Expected 2D actions, got {dataset.actions.ndim}D"


run_test("Dataset loading (D4RLDataset)", test_dataset_loading)


# ── Test 3: Normalization ────────────────────────────────────────────────────

def test_normalization():
    global dataset
    if dataset is None:
        raise RuntimeError("Skipped — dataset not loaded")

    from iql.dataset_utils import split_into_trajectories
    import numpy as np

    # Reproduce the normalize() function from the notebook
    trajs = split_into_trajectories(
        dataset.observations, dataset.actions, dataset.rewards,
        dataset.masks, dataset.dones_float, dataset.next_observations)

    def compute_returns(traj):
        return sum(rew for _, _, rew, _, _, _ in traj)

    trajs.sort(key=compute_returns)
    ret_range = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    print(f"  Number of trajectories: {len(trajs)}")
    print(f"  Return range: {ret_range:.2f}")

    assert abs(ret_range) > 1e-8, "Return range is ~0, normalization would fail"

    reward_before = dataset.rewards.mean()
    dataset.rewards /= ret_range
    dataset.rewards *= 1000.0
    reward_after = dataset.rewards.mean()
    print(f"  Reward mean before: {reward_before:.4f}")
    print(f"  Reward mean after:  {reward_after:.4f}")


run_test("Normalization", test_normalization)


# ── Test 4: Environment creation ─────────────────────────────────────────────

def test_env_creation():
    import gymnasium as gym
    import wrappers
    from iql.dataset_utils import d4rl_to_gymnasium_name

    env_name = 'hopper-medium-v2'
    gym_env_name = d4rl_to_gymnasium_name(env_name)
    print(f"  D4RL name: {env_name} -> Gymnasium name: {gym_env_name}")

    env = gym.make(gym_env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    obs = env.reset(seed=42)
    # gymnasium returns (obs, info)
    if isinstance(obs, tuple):
        obs = obs[0]
    print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Action space: {env.action_space}")
    assert obs.dtype.name == 'float32', f"Expected float32, got {obs.dtype}"
    env.close()
    print("  Environment created and reset successfully")


run_test("Environment creation", test_env_creation)


# ── Test 5: Learner initialization ───────────────────────────────────────────

agent_2q = None
agent_3q = None

def test_learner_init():
    global agent_2q, agent_3q
    import numpy as np
    import gymnasium as gym
    import wrappers
    from iql import Learner
    from iql.dataset_utils import d4rl_to_gymnasium_name

    gym_env_name = d4rl_to_gymnasium_name('hopper-medium-v2')
    env = gym.make(gym_env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.reset(seed=42)

    CONFIG = {
        'actor_lr': 3e-4, 'value_lr': 3e-4, 'critic_lr': 3e-4,
        'hidden_dims': (256, 256), 'discount': 0.99,
        'expectile': 0.7, 'temperature': 3.0,
        'dropout_rate': None, 'tau': 0.005,
    }

    obs_sample = env.observation_space.sample()[np.newaxis]
    act_sample = env.action_space.sample()[np.newaxis]

    print("  Creating 2Q Learner...")
    agent_2q = Learner(42, obs_sample, act_sample,
                       max_steps=10, num_critics=2, **CONFIG)
    print(f"  ✅ 2Q Learner created")

    print("  Creating 3Q Learner...")
    agent_3q = Learner(42, obs_sample, act_sample,
                       max_steps=10, num_critics=3, **CONFIG)
    print(f"  ✅ 3Q Learner created")

    env.close()


run_test("Learner initialization (2Q + 3Q)", test_learner_init)


# ── Test 6: Training step ────────────────────────────────────────────────────

def test_training():
    global agent_2q, agent_3q, dataset
    if agent_2q is None or agent_3q is None:
        raise RuntimeError("Skipped — learners not initialized")
    if dataset is None:
        raise RuntimeError("Skipped — dataset not loaded")

    BATCH_SIZE = 256
    TRAIN_STEPS = 10

    print(f"  Training 2Q for {TRAIN_STEPS} steps...")
    for i in range(1, TRAIN_STEPS + 1):
        batch = dataset.sample(BATCH_SIZE)
        info = agent_2q.update(batch)
    print(f"  ✅ 2Q: {TRAIN_STEPS} steps completed. Last info keys: {list(info.keys())}")

    print(f"  Training 3Q for {TRAIN_STEPS} steps...")
    for i in range(1, TRAIN_STEPS + 1):
        batch = dataset.sample(BATCH_SIZE)
        info = agent_3q.update(batch)
    print(f"  ✅ 3Q: {TRAIN_STEPS} steps completed. Last info keys: {list(info.keys())}")


run_test("Training step (10 steps each)", test_training)


# ── Test 7: Evaluation ──────────────────────────────────────────────────────

def test_evaluation():
    global agent_2q
    if agent_2q is None:
        raise RuntimeError("Skipped — learner not initialized")

    import gymnasium as gym
    import wrappers
    from iql.dataset_utils import d4rl_to_gymnasium_name
    from evaluation import evaluate

    gym_env_name = d4rl_to_gymnasium_name('hopper-medium-v2')
    env = gym.make(gym_env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.reset(seed=42)
    env.action_space.seed(42)

    print("  Running evaluate(agent_2q, env, 2)...")
    stats = evaluate(agent_2q, env, 2)
    print(f"  Return: {stats['return']:.2f}")
    print(f"  Length: {stats['length']:.0f}")
    assert 'return' in stats, "Missing 'return' key in stats"
    assert 'length' in stats, "Missing 'length' key in stats"
    env.close()


run_test("Evaluation (2 episodes)", test_evaluation)


# ── Test 8: Shift wrappers ──────────────────────────────────────────────────

def test_shift_wrappers():
    global agent_2q
    if agent_2q is None:
        raise RuntimeError("Skipped — learner not initialized")

    import gymnasium as gym
    import wrappers
    from iql.dataset_utils import d4rl_to_gymnasium_name
    from evaluation import evaluate
    from wrappers.gravity_shift import GravityShift
    from wrappers.observation_noise import ObservationNoise
    from wrappers.friction_shift import FrictionShift
    from wrappers.reward_perturbation import RewardPerturbation

    gym_env_name = d4rl_to_gymnasium_name('hopper-medium-v2')

    wrapper_configs = [
        ("GravityShift", GravityShift, {'gravity_scale': 1.5}),
        ("ObservationNoise", ObservationNoise, {'noise_std': 0.1}),
        ("FrictionShift", FrictionShift, {'friction_scale': 1.5}),
        ("RewardPerturbation", RewardPerturbation, {'noise_std': 0.1}),
    ]

    for name, WrapperCls, kwargs in wrapper_configs:
        env = gym.make(gym_env_name)
        env = WrapperCls(env, **kwargs)
        env = wrappers.EpisodeMonitor(env)
        env = wrappers.SinglePrecision(env)
        env.reset(seed=42)
        env.action_space.seed(42)

        stats = evaluate(agent_2q, env, 1)  # 1 episode per wrapper
        print(f"  {name}({kwargs}): return={stats['return']:.2f}, length={stats['length']:.0f}")
        env.close()


run_test("Shift wrappers (4 types)", test_shift_wrappers)


# ── Test 9: Pre-computed CSV loading ─────────────────────────────────────────

def test_csv_loading():
    import csv

    csv_files = [
        'results/shift_hopper-medium-v2_2Q_seed42.csv',
        'results/shift_hopper-medium-v2_3Q_seed42.csv',
    ]

    for filepath in csv_files:
        full_path = os.path.join(PROJECT_ROOT, filepath)
        assert os.path.exists(full_path), f"CSV not found: {full_path}"

        rows = []
        with open(full_path) as f:
            for row in csv.DictReader(f):
                for k in row:
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass
                rows.append(row)

        print(f"  {filepath}: {len(rows)} rows")
        assert len(rows) > 0, f"CSV is empty: {filepath}"

        # Verify expected columns
        expected_cols = {'shift_type', 'shift_level', 'raw_return', 'episode_length', 'robustness_drop'}
        actual_cols = set(rows[0].keys())
        assert expected_cols.issubset(actual_cols), \
            f"Missing columns: {expected_cols - actual_cols}"

        # Verify shift types present
        shift_types = set(r['shift_type'] for r in rows)
        print(f"    Shift types: {shift_types}")
        assert len(shift_types) == 4, f"Expected 4 shift types, got {len(shift_types)}"


run_test("Pre-computed CSV loading", test_csv_loading)


# ── Test 10: Matplotlib plotting ─────────────────────────────────────────────

def test_matplotlib():
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    x = [1, 2, 3, 4]
    y1 = [100, 90, 70, 50]
    y2 = [100, 95, 80, 60]
    ax.plot(x, y1, '-o', label='Baseline (2Q)', color='steelblue')
    ax.plot(x, y2, '--s', label='Q-Ensemble (3Q)', color='coral')
    ax.set_xlabel('Shift Level')
    ax.set_ylabel('Average Return')
    ax.set_title('Test Plot — Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, 'scripts', 'test_plot.png')
    plt.savefig(out_path, dpi=72)
    plt.close()

    assert os.path.exists(out_path), f"Plot file not created: {out_path}"
    file_size = os.path.getsize(out_path)
    print(f"  Saved test plot: {out_path} ({file_size:,} bytes)")
    assert file_size > 1000, f"Plot file suspiciously small: {file_size} bytes"

    # Clean up
    os.remove(out_path)
    print("  Cleaned up test plot")


run_test("Matplotlib plotting", test_matplotlib)


# ── Summary ──────────────────────────────────────────────────────────────────

total_elapsed = time.time() - TOTAL_START
print(f"\n\n{'='*60}")
print(f"VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"Total time: {total_elapsed:.1f}s\n")

passed = 0
failed = 0
for name, (status, elapsed, tb) in RESULTS.items():
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon} {name} ({elapsed:.1f}s)")
    if status == "PASS":
        passed += 1
    else:
        failed += 1

print(f"\n  {passed} passed, {failed} failed out of {passed + failed} tests")

if failed > 0:
    print("\n  FAILED TESTS:")
    for name, (status, elapsed, tb) in RESULTS.items():
        if status == "FAIL":
            print(f"\n  --- {name} ---")
            print(f"  {tb}")

sys.exit(1 if failed > 0 else 0)

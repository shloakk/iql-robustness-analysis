#!/bin/bash
#SBATCH --job-name=iql_robustness
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#
# IQL Robustness Analysis — SJSU CoE HPC Experiment Runner
#
# FIRST TIME SETUP (run once on the login node, NOT as a batch job):
#   bash scripts/run_all_hpc.sh setup
#
# Then submit experiments:
#   sbatch scripts/run_all_hpc.sh            # full pipeline
#   sbatch scripts/run_all_hpc.sh train      # training only
#   sbatch scripts/run_all_hpc.sh eval       # evaluation only
#   sbatch scripts/run_all_hpc.sh analyze    # analysis only
#
# The setup step installs Miniconda (Python 3.11) in your home directory
# and creates a conda env with all dependencies. This only needs to be
# done once. GPU nodes share /home so the env is available everywhere.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
ENV_NAME="iql"
PYTHON_VERSION="3.11"
ENVIRONMENTS="hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2"
CRITIC_CONFIGS="2 3"
MAX_STEPS=300000
SEEDS="42"
TAU_VALUES="0.5 0.8 0.9"

# ─────────────────────────────────────────────────────────────────────
# PROJECT DIR
# ─────────────────────────────────────────────────────────────────────
if [ -f "scripts/run_all_hpc.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "run_all_hpc.sh" ]; then
    PROJECT_DIR="$(cd .. && pwd)"
else
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
fi
cd "$PROJECT_DIR"

mkdir -p logs results tmp

echo "============================================"
echo "IQL Robustness Analysis — HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# CONDA INIT HELPER
# ─────────────────────────────────────────────────────────────────────
init_conda() {
    # Try to find and initialize conda
    for CONDA_DIR in ~/miniconda3 ~/anaconda3; do
        if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_DIR}/etc/profile.d/conda.sh"
            return 0
        fi
    done
    return 1
}

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT ACTIVATION (used by batch jobs)
# ─────────────────────────────────────────────────────────────────────
activate_env() {
    module load cuda 2>/dev/null || true

    if ! init_conda; then
        echo "ERROR: Conda not found. Run setup first on the login node:"
        echo "  bash scripts/run_all_hpc.sh setup"
        exit 1
    fi

    if ! conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        echo "ERROR: Conda env '${ENV_NAME}' not found. Run setup first:"
        echo "  bash scripts/run_all_hpc.sh setup"
        exit 1
    fi

    conda activate "${ENV_NAME}" 2>/dev/null || source activate "${ENV_NAME}"
    echo "Activated conda env: ${ENV_NAME}"
    python --version
}

# ─────────────────────────────────────────────────────────────────────
# STEP 0: One-time setup (run on login node — needs internet)
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> One-time environment setup"
    echo ">>> Run this on the LOGIN NODE (not as sbatch)"
    echo ""

    # Step 1: Install Miniconda if not present
    if ! init_conda; then
        echo "Conda not found. Installing Miniconda..."
        INSTALLER="/tmp/miniconda_installer_$$.sh"
        wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
            -O "$INSTALLER"
        bash "$INSTALLER" -b -p ~/miniconda3
        rm -f "$INSTALLER"
        source ~/miniconda3/etc/profile.d/conda.sh
        conda init bash 2>/dev/null || true
        echo ""
        echo "Miniconda installed at ~/miniconda3"
        echo "You may need to log out and back in, or run:"
        echo "  source ~/miniconda3/etc/profile.d/conda.sh"
        echo ""
    else
        echo "Conda found: $(conda --version)"
    fi

    # Step 2: Create conda env with Python 3.11
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo "Creating conda environment: ${ENV_NAME} (python ${PYTHON_VERSION})"
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    else
        echo "Conda env '${ENV_NAME}' already exists"
    fi

    # Step 3: Activate and install deps
    conda activate "${ENV_NAME}" 2>/dev/null || source activate "${ENV_NAME}"
    echo "Python: $(python --version)"
    echo "pip:    $(pip --version)"

    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install jax jaxlib flax optax
    pip install mujoco "gymnasium[mujoco]" gym
    pip install h5py tqdm matplotlib numpy scipy
    pip install absl-py ml_collections tensorboardX tensorflow-probability

    echo ""
    echo "Installing D4RL..."
    pip install git+https://github.com/Farama-Foundation/d4rl@master 2>/dev/null || \
        echo "WARNING: D4RL install had issues (may still work)"

    # Verify
    echo ""
    echo "Verifying installation..."
    python -c "
import sys
print(f'Python:     {sys.version.split()[0]}')
import jax
print(f'JAX:        {jax.__version__}')
import flax
print(f'Flax:       {flax.__version__}')
import gymnasium
print(f'Gymnasium:  {gymnasium.__version__}')
print('All dependencies installed successfully.')
"

    echo ""
    echo "============================================"
    echo "Setup complete. Now submit a batch job:"
    echo "  sbatch scripts/run_all_hpc.sh"
    echo "  sbatch scripts/run_all_hpc.sh train"
    echo "============================================"
}

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Training (3 envs x 2 critic configs = 6 runs)
# ─────────────────────────────────────────────────────────────────────
run_training() {
    echo ""
    echo ">>> Phase 1: Training"
    echo "    Environments: $ENVIRONMENTS"
    echo "    Critics:      $CRITIC_CONFIGS"
    echo "    Steps:        $MAX_STEPS"
    echo ""

    for env in $ENVIRONMENTS; do
        for nq in $CRITIC_CONFIGS; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"
                echo "--- Training: ${env} | ${nq}Q | seed=${seed} ---"
                python scripts/train_offline.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --num_critics="${nq}" \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}"
                echo "    Saved to ${SAVE_DIR}"
            done
        done
    done

    echo "Training complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Shift evaluation (all 4 shift types per model)
# ─────────────────────────────────────────────────────────────────────
run_evaluation() {
    echo ""
    echo ">>> Phase 2: Shift Evaluation"
    echo ""

    for env in $ENVIRONMENTS; do
        for nq in $CRITIC_CONFIGS; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/${env}_${nq}Q_s${seed}"
                echo "--- Evaluating: ${env} | ${nq}Q | seed=${seed} ---"
                python scripts/evaluate_shift.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --num_critics="${nq}" \
                    --shift_type=all \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}" \
                    --output_dir=results/
                echo "    Results written to results/"
            done
        done
    done

    echo "Shift evaluation complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Expectile tau ablation (2Q only, 3 tau values)
# ─────────────────────────────────────────────────────────────────────
run_ablation() {
    echo ""
    echo ">>> Phase 3: Expectile tau Ablation"
    echo "    Tau values: $TAU_VALUES"
    echo ""

    mkdir -p results/ablation_tau

    for env in $ENVIRONMENTS; do
        for tau in $TAU_VALUES; do
            for seed in $SEEDS; do
                SAVE_DIR="tmp/abl_tau${tau}_${env}_s${seed}"
                echo "--- Ablation: ${env} | tau=${tau} | seed=${seed} ---"

                python scripts/train_offline.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --config.expectile="${tau}" \
                    --num_critics=2 \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}"

                python scripts/evaluate_shift.py \
                    --env_name="${env}" \
                    --config=configs/mujoco_config.py \
                    --config.expectile="${tau}" \
                    --num_critics=2 \
                    --shift_type=all \
                    --max_steps="${MAX_STEPS}" \
                    --seed="${seed}" \
                    --save_dir="${SAVE_DIR}" \
                    --output_dir=results/ablation_tau/
            done
        done
    done

    echo "Ablation complete."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Analysis (compute robustness metrics)
# ─────────────────────────────────────────────────────────────────────
run_analysis() {
    echo ""
    echo ">>> Phase 4: Analysis"
    echo ""

    for env in $ENVIRONMENTS; do
        echo "--- Metrics: ${env} ---"
        python scripts/compute_robustness.py \
            --results_dir=results/ \
            --env_name="${env}"
    done

    echo ""
    echo "Analysis complete. Results in results/"
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    setup)
        setup_environment
        ;;
    train)
        activate_env
        run_training
        ;;
    eval)
        activate_env
        run_evaluation
        ;;
    ablation)
        activate_env
        run_ablation
        ;;
    analyze)
        activate_env
        run_analysis
        ;;
    all)
        activate_env
        run_training
        run_evaluation
        run_ablation
        run_analysis
        ;;
    *)
        echo "Usage:"
        echo "  First time:  bash scripts/run_all_hpc.sh setup    (on login node)"
        echo "  Then:        sbatch scripts/run_all_hpc.sh         (full pipeline)"
        echo "               sbatch scripts/run_all_hpc.sh train"
        echo "               sbatch scripts/run_all_hpc.sh eval"
        echo "               sbatch scripts/run_all_hpc.sh ablation"
        echo "               sbatch scripts/run_all_hpc.sh analyze"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo ""
echo "Results:"
ls -la results/ 2>/dev/null || echo "  (no results directory)"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r $(whoami)@coe-hpc1.sjsu.edu:${PROJECT_DIR}/results/ ./results/"
echo "============================================"

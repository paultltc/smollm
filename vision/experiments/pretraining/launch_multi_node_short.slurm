#!/bin/bash
#SBATCH -A nwd@h100
#SBATCH -C h100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=14:00:00
#SBATCH --job-name=modality_align
#SBATCH --mem-bind=local
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/nwd/uyn61im/logs/modality_alignment/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/nwd/uyn61im/logs/modality_alignment/%x_%j.err

set -euo pipefail

# Get the config file from the command line
while getopts "c:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;
    *) echo "Usage: $0 -c <config_file>"; exit 1 ;;
  esac
done
echo "Using config: ${CONFIG_FILE}"

# Load the environment
mkdir -p ${SCRATCH}/logs/modality_alignment
module purge
module load arch/h100 cuda/12.4.1
source .venv/bin/activate

# Cluster-specific settings
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

# We are on an offline partition
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
wandb offline

# Load the conda environment
export HF_HOME=$SCRATCH/hf_home
export ACCELERATE_HOME=$HF_HOME/accelerate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # dynamic memory allocation, we have varying sequence lengths

echo "launching script"

MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
MASTER_PORT=6001 # Most of the ports are available (and if not, there will be an error thrown out)
ACCELERATE_CONFIG_FILE="${ACCELERATE_HOME}/zero2_multi_node.yaml"

export SRUN_ARGS=" \
    --label \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

export LAUNCHER="PYTHONPATH=$(pwd) accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    "

export PROGRAM=" \
    m4/training/main.py \
        --config $CONFIG_FILE \
    "

export CMD="$LAUNCHER $PROGRAM"

srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a ${SCRATCH}/logs/modality_alignment/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
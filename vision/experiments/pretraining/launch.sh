#!/bin/bash

CONFIG_FILE=""

while getopts "c:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;
    *) echo "Usage: $0 -c <config_file>"; exit 1 ;;
  esac
done

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

# be careful about the cache folder for Wandb
wandb offline

# Load the environment
export HF_HOME=$SCRATCH/hf_home
export ACCELERATE_HOME=$HF_HOME/accelerate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCELERATE_CONFIG_FILE="${ACCELERATE_HOME}/zero1_single_gpu.yaml"
MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
MASTER_PORT=8000

PYTHONPATH=$(pwd) accelerate launch \
        --config_file $ACCELERATE_CONFIG_FILE \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        m4/training/main.py \
          --config $CONFIG_FILE
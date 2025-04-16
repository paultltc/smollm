#!/bin/bash
#SBATCH -A nwd@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --job-name=modality_align
#SBATCH --mem-bind=local
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/nwd/uyn61im/logs/modality_alignment/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/nwd/uyn61im/logs/modality_alignment/%x_%j.err

set -euo pipefail

mkdir -p ${SCRATCH}/logs/modality_alignment

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

module purge
module load arch/h100 cuda/12.4.1

export HF_DATASETS_OFFLINE=1
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_ENTITY="smolvencoder"
export WANDB_PROJECT="modality_alignment"

source $WORK/venv_h100/bin/activate
wandb offline
echo "launching script"

while getopts "c:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;
    *) echo "Usage: $0 -c <config_file>"; exit 1 ;;
  esac
done

echo "Using config: ${CONFIG_FILE}"

ACCELERATE_CONFIG_FILE=$HF_HOME/accelerate/default_config.yaml
DEEPSPEED_CONFIG_FILE=$HF_HOME/deepspeed/default_config.json

SRUN_ARGS=" \
    --label \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

CMD="PYTHONPATH=$(pwd) accelerate launch \
        --config_file ${ACCELERATE_CONFIG_FILE} \
        m4/training/main.py \
        --config ${CONFIG_FILE}"

srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a ${SCRATCH}/logs/modality_alignment/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

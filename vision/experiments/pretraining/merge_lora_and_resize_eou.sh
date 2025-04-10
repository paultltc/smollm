#!/bin/bash

HOME_DIR="/home/paulteiletche"
CONFIG_FILE=""

while getopts "c:" opt; do
  case "$opt" in
    c) OPT_STEP_DIR="$OPTARG" ;;
    *) echo "Usage: $0 -c <config_file>"; exit 1 ;;
  esac
done


OPT_STEP_DIR="${HOME_DIR}/smolvencoder/vision/${OPT_STEP_DIR}"
OUTPUT_DIR="${OPT_STEP_DIR}__merge_and_resize_eou"

PYTHONPATH=$(pwd) python m4/scripts/merge_lora_and_save.py $OPT_STEP_DIR $OUTPUT_DIR
# PYTHONPATH=$(pwd) python vision/experiments/pretraining/resize_embed_for_eou.py $OUTPUT_DIR
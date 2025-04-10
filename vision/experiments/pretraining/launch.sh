#!/bin/bash

HOME_DIR="/home/paulteiletche"
ACCELERATE_CONFIG_FILE="${HOME_DIR}/.cache/huggingface/accelerate/default_config.yaml"
CONFIG_FILE=""

while getopts "c:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;
    *) echo "Usage: $0 -c <config_file>"; exit 1 ;;
  esac
done

CONFIG_FILE="${HOME_DIR}/smolvencoder/vision/${CONFIG_FILE}"

PYTHONPATH=$(pwd) python -u -m accelerate.commands.launch \
    --config_file $ACCELERATE_CONFIG_FILE \
        m4/training/main.py \
        --config $CONFIG_FILE
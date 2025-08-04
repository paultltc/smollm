#!/bin/bash

# Defaults
RUN_NAME=""
PROJECT="modality_align"
ENTITY="smolvencoder"
BASE_DIR="wandb"

# Help message
usage() {
    echo "Usage: $0 [--run <run_name>] [--project <project_name>] [--entity <entity_name>]"
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --run)
            RUN_NAME="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check base dir exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' does not exist."
    exit 1
fi

# Determine most recent run if not specified
if [ -z "$RUN_NAME" ]; then
    RUN_NAME=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d | sort -r | head -n 1 | xargs basename)
    if [ -z "$RUN_NAME" ]; then
        echo "Error: No valid run folders found in '$BASE_DIR/'"
        exit 1
    fi
fi

# Full path
RUN_PATH="$BASE_DIR/$RUN_NAME"

# Check run folder exists
if [ ! -d "$RUN_PATH" ]; then
    echo "Error: Run folder '$RUN_PATH' does not exist or is not a directory."
    exit 1
fi

# Run sync
echo "Syncing run: $RUN_PATH"
echo "Project: $PROJECT"
echo "Entity: $ENTITY"
watch -n 10 wandb sync -p "$PROJECT" -e "$ENTITY" "$RUN_PATH"

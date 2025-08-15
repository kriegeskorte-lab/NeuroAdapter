#!/usr/bin/env bash
set -euo pipefail

#
# Multi-Subject Brain Adapter Training and Evaluation Pipeline
#
# This script runs the complete pipeline for multi-subject brain adapter training:
# 1. Train brain adapter on multiple subjects using shared parcel selection
# 2. Decode/generate images for a specific test subject  
# 3. Generate comparison plots
# 4. Compute evaluation metrics
#
# The script creates organized output directories with subject-specific subdirectories
# for decoded stimuli, plots, and metrics when using multi-subject training.
#

# Configuration
TIME=$(date +%m_%d_%Y-%H_%M)
EPOCHS=100
TRAIN_BATCH_SIZE=16
TOPK=100
NUM_DECODER_QUERIES=50
CONDITION_DIM=768
# SUB_APPROACH="transformer_decoder"
SUB_APPROACH="linear_projection"
TRAINING_SUBJECT_IDS="1 2 3 4 5 6 7 8"
TESTING_SUBJECT_ID=1
START_IDX=0
END_IDX=8
NUM_PREDICTIONS=4
LR=1e-4
NUM_GPUS=4

# Setup
LOG_DIR="logs/${TIME}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/pipeline.log"
TRAIN_LOG="${LOG_DIR}/training.log"
DECODE_LOG="${LOG_DIR}/decoding.log"
METRICS_LOG="${LOG_DIR}/metrics.log"
PLOT_LOG="${LOG_DIR}/plotting.log"

# Simple logging
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

echo "Brain Adapter Multi-Subject Pipeline Started"
echo "Training subjects: $TRAINING_SUBJECT_IDS"
echo "Testing subject: $TESTING_SUBJECT_ID"
echo "Logs: $LOG_DIR"
echo ""

# Save config
echo "TIME=$TIME" > "${LOG_DIR}/config.txt"
echo "EPOCHS=$EPOCHS" >> "${LOG_DIR}/config.txt"
echo "TESTING_SUBJECT_ID=$TESTING_SUBJECT_ID" >> "${LOG_DIR}/config.txt"
echo "TRAINING_SUBJECT_IDS=$TRAINING_SUBJECT_IDS" >> "${LOG_DIR}/config.txt"
echo "TOPK=$TOPK" >> "${LOG_DIR}/config.txt"
echo "SUB_APPROACH=$SUB_APPROACH" >> "${LOG_DIR}/config.txt"

# Steps
log "Starting training..."
accelerate launch --config_file acc_config.yaml --num_processes $NUM_GPUS train_brain_adapter.py \
    --time "$TIME" --learning_rate $LR --num_train_epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE --dataloader_num_workers 8 \
    --training_subjects $TRAINING_SUBJECT_IDS --topk $TOPK --condition_dim $CONDITION_DIM \
    --num_decoder_queries $NUM_DECODER_QUERIES --sub_approach $SUB_APPROACH \
    --wandb 2>&1 | tee "$TRAIN_LOG"

log "Training completed"

# Check if training was successful
if [ ! -f "brain_adapter/model_weights/$TIME/checkpoint-$EPOCHS/pytorch_model.bin" ]; then
    log "ERROR: Training checkpoint not found. Aborting pipeline."
    exit 1
fi

log "Starting decoding..."
python decode_brain_adapter.py \
    --model_weights_dir "brain_adapter/model_weights/$TIME" \
    --saved_epochs $EPOCHS --save_all_candidates \
    --start_idx $START_IDX --end_idx $END_IDX \
    --num_predictions $NUM_PREDICTIONS --subject_id $TESTING_SUBJECT_ID \
    --topk $TOPK --num_decoder_queries $NUM_DECODER_QUERIES \
    --condition_dim $CONDITION_DIM --sub_approach $SUB_APPROACH \
    --multi_subject_training \
    2>&1 | tee "$DECODE_LOG"

python decode_brain_adapter.py \
    --model_weights_dir "brain_adapter/model_weights/$TIME" \
    --saved_epochs $EPOCHS --eval_full_dataset \
    --num_predictions $NUM_PREDICTIONS --subject_id $TESTING_SUBJECT_ID \
    --topk $TOPK --num_decoder_queries $NUM_DECODER_QUERIES \
    --condition_dim $CONDITION_DIM --sub_approach $SUB_APPROACH \
    --multi_subject_training \
    2>&1 | tee -a "$DECODE_LOG"

log "Decoding completed"

# Check if decoding was successful
if [ ! -f "brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/subset/$TESTING_SUBJECT_ID/evaluation_metadata.json" ]; then
    log "WARNING: Subset decoding results not found"
fi

if [ ! -f "brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/full/$TESTING_SUBJECT_ID/evaluation_metadata.json" ]; then
    log "WARNING: Full decoding results not found"
fi


log "Generating plots..."
python plot_brain_adapter.py \
    --results_dir "brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/subset/$TESTING_SUBJECT_ID" \
    --start_idx $START_IDX --end_idx $END_IDX \
    2>&1 | tee "$PLOT_LOG"

log "Plotting completed"

log "Computing metrics..."
python metric_brain_adapter.py \
    --results_dir "brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/full/$TESTING_SUBJECT_ID" \
    --saved_epochs $EPOCHS --evaluation_mode full \
    2>&1 | tee "$METRICS_LOG"

log "Metrics completed"

# Summary
echo "Duration: $SECONDS seconds" > "${LOG_DIR}/summary.txt"
echo "Model weights: brain_adapter/model_weights/$TIME" >> "${LOG_DIR}/summary.txt"
echo "Results: brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/subset/$TESTING_SUBJECT_ID" >> "${LOG_DIR}/summary.txt"
echo "Plots: brain_adapter/plotted_stimuli/$TIME/epoch_$EPOCHS/subset/$TESTING_SUBJECT_ID" >> "${LOG_DIR}/summary.txt"

echo ""
echo "Pipeline completed!"
echo "Check: $LOG_DIR"
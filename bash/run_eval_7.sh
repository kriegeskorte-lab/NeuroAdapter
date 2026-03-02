#!/usr/bin/env bash
set -euo pipefail

# Configuration
# TIME=$(date +%m_%d_%Y-%H_%M)
# TIME="09_13_2025-14_59"
# EPOCHS=200
TIME="09_17_2025-13_38"
EPOCHS=300
TRAIN_BATCH_SIZE=16
TOPK=100
NUM_DECODER_QUERIES=50
CONDITION_DIM=768
# SUB_APPROACH="transformer_decoder"
SUB_APPROACH="linear_projection"
SUBJECT_ID=7
START_IDX=0
END_IDX=8
NOISE_FACTOR=5.0
NUM_PREDICTIONS=7
LR=1e-4

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

echo "Brain Adapter Pipeline Started"
echo "Logs: $LOG_DIR"

# Save config
echo "TIME=$TIME" > "${LOG_DIR}/config.txt"
echo "EPOCHS=$EPOCHS" >> "${LOG_DIR}/config.txt"
echo "SUBJECT_ID=$SUBJECT_ID" >> "${LOG_DIR}/config.txt"

# python decode_brain_adapter_memory.py \
python decode_brain_adapter.py \
    --model_weights_dir "brain_adapter/model_weights/$TIME" \
    --saved_epochs $EPOCHS --eval_full_dataset \
    --num_predictions $NUM_PREDICTIONS --subject_id $SUBJECT_ID \
    --noise_factor $NOISE_FACTOR --topk $TOPK \
    --num_decoder_queries $NUM_DECODER_QUERIES \
    --condition_dim $CONDITION_DIM --sub_approach $SUB_APPROACH \
    2>&1 | tee "$DECODE_LOG"

log "Decoding completed"

log "Computing metrics..."
python metric_brain_adapter.py \
    --model_weights_dir "brain_adapter/model_weights/$TIME" \
    --saved_epochs $EPOCHS --evaluation_mode full \
    2>&1 | tee "$METRICS_LOG"

log "Metrics completed"

# Summary
echo "Duration: $SECONDS seconds" > "${LOG_DIR}/summary.txt"
echo "Model weights: brain_adapter/model_weights/$TIME" >> "${LOG_DIR}/summary.txt"
echo "Results: brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/subset" >> "${LOG_DIR}/summary.txt"

echo ""
echo "Pipeline completed!"
echo "Check: $LOG_DIR"


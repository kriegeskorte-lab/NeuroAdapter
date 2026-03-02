#!/usr/bin/env bash
set -euo pipefail

# Configuration
# TIME=$(date +%m_%d_%Y-%H_%M)
TIME="09_25_2025-14_54"
EPOCHS=500
TRAIN_BATCH_SIZE=16
TOPK=100
NUM_DECODER_QUERIES=50
CONDITION_DIM=768
# SUB_APPROACH="transformer_decoder"
SUB_APPROACH="linear_projection"
SUBJECT_ID=1
START_IDX=0
END_IDX=8
NOISE_FACTOR=4.0
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

# log "Starting decoding subset..."
# python decode_brain_adapter.py \
#     --model_weights_dir "brain_adapter/model_weights/$TIME" \
#     --saved_epochs $EPOCHS --save_all_candidates \
#     --start_idx $START_IDX --end_idx $END_IDX \
#     --num_predictions $NUM_PREDICTIONS --subject_id $SUBJECT_ID \
#     --noise_factor $NOISE_FACTOR --topk $TOPK \
#     --num_decoder_queries $NUM_DECODER_QUERIES \
#     --condition_dim $CONDITION_DIM --sub_approach $SUB_APPROACH \
#     2>&1 | tee "$DECODE_LOG"

# log "Decoding subset completed"

# log "Generating subset plots..."
# python plot_brain_adapter.py \
#     --results_dir "brain_adapter/decoded_stimuli/$TIME/epoch_$EPOCHS/subset" \
#     --start_idx $START_IDX --end_idx $END_IDX \
#     2>&1 | tee "$PLOT_LOG"

# log "Plotting subset completed"

python decode_brain_adapter_memory.py \
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
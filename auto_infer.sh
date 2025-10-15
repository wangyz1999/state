#!/bin/bash

# Automated inference script - Configuration at top
# Add your checkpoint paths to the CHECKPOINT_PATHS array below

# =============================================================================
# CONFIGURATION - Edit the settings here
# =============================================================================

# CUDA device to use (e.g., 0, 1, 2, or "0,1" for multiple GPUs)
CUDA_DEVICE="4"

USE_EMB=False

# Checkpoint paths to process
CHECKPOINT_PATHS=(
    "/home/wyunzhe/projects/state/competition/state_md_full_dataset/checkpoints/step=step=8000-val_loss=val_loss=0.8757.ckpt"
    "/home/wyunzhe/projects/state/competition/state_sm_full_dataset/checkpoints/step=step=20000-val_loss=val_loss=0.7973.ckpt"
    "/home/wyunzhe/projects/state/competition/state_sm_tahoe_full_dataset/checkpoints/step=step=10000-val_loss=val_loss=2.0163.ckpt"
)

# =============================================================================
# SCRIPT EXECUTION - No need to modify below this line
# =============================================================================

if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "Error: No checkpoint paths configured!"
    echo "Please add checkpoint paths to the CHECKPOINT_PATHS array at the top of this script."
    exit 1
fi

echo "=== Processing ${#CHECKPOINT_PATHS[@]} checkpoint(s) ==="
echo ""

# Array to track generated .prep.vcc files
PREP_VCC_FILES=()

# Process each checkpoint path
for i in "${!CHECKPOINT_PATHS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_PATHS[$i]}"
    
    echo "=== Processing checkpoint $((i+1))/${#CHECKPOINT_PATHS[@]} ==="
    echo "Checkpoint: $CHECKPOINT_PATH"
    
    # Extract model directory (parent of checkpoints folder)
    MODEL_DIR=$(dirname $(dirname "$CHECKPOINT_PATH"))
    
    # Extract checkpoint filename without extension for output naming
    CHECKPOINT_FILENAME=$(basename "$CHECKPOINT_PATH")
    CHECKPOINT_NAME="${CHECKPOINT_FILENAME%.*}"
    
    # Generate output filename
    OUTPUT_FILE="${MODEL_DIR}/${CHECKPOINT_NAME}_pred.h5ad"
    
    echo "Model Dir: $MODEL_DIR"
    echo "Output: $OUTPUT_FILE"
    echo "=================================="
    echo "Using embedding: $USE_EMB"
    echo "=================================="
    # Step 1: Run inference
    echo "Step 1: Running inference on CUDA device(s): $CUDA_DEVICE"
    if $USE_EMB; then
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE uv run state tx infer \
        --output "$OUTPUT_FILE" \
        --embed_key "X_state"
        --model_dir "$MODEL_DIR" \
        --checkpoint "$CHECKPOINT_PATH" \
        --adata "data_state_emb/competition_val_template.h5ad" \
        --pert_col "target_gene"
    else
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE uv run state tx infer \
        --output "$OUTPUT_FILE" \
        --model_dir "$MODEL_DIR" \
        --checkpoint "$CHECKPOINT_PATH" \
        --adata "data_state_emb/competition_val_template.h5ad" \
        --pert_col "target_gene"
    fi
    
    # Check if inference was successful
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for checkpoint $((i+1))!"
        echo "Skipping to next checkpoint..."
        echo ""
        continue
    fi
    
    echo "Step 1 completed successfully!"
    
    # Step 2: Run cell-eval prep
    echo "Step 2: Running cell-eval prep..."
    cell-eval prep -i "$OUTPUT_FILE" -g "competition_support_set/gene_names.csv"
    
    # Check if prep was successful
    if [ $? -ne 0 ]; then
        echo "Error: cell-eval prep failed for checkpoint $((i+1))!"
        echo "Skipping to next checkpoint..."
        echo ""
        continue
    fi
    
    echo "Step 2 completed successfully!"
    
    # Track the generated .prep.vcc file (cell-eval adds .prep.vcc to the filename without .h5ad)
    OUTPUT_FILE_BASE="${OUTPUT_FILE%.h5ad}"  # Remove .h5ad extension
    PREP_VCC_FILE="${OUTPUT_FILE_BASE}.prep.vcc"
    PREP_VCC_FILES+=("$(realpath "$PREP_VCC_FILE")")
    
    echo "=== Checkpoint $((i+1)) completed successfully! ==="
    echo "Output file: $OUTPUT_FILE"
    echo "Prep file: $PREP_VCC_FILE"
    echo ""
done

echo "=== All checkpoints processed! ==="
echo ""

# Print all generated .prep.vcc files
if [ ${#PREP_VCC_FILES[@]} -gt 0 ]; then
    echo "=== Generated .prep.vcc files ==="
    for prep_file in "${PREP_VCC_FILES[@]}"; do
        echo "$prep_file"
    done
    echo ""
    echo "Total files generated: ${#PREP_VCC_FILES[@]}"
else
    echo "No .prep.vcc files were generated (all checkpoints failed)"
fi

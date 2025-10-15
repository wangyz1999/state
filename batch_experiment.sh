#!/usr/bin/env bash
# Grid launcher for state tx train
# Runs one sequential worker per GPU so no two jobs share a GPU.

########################################
# User settings
########################################

# List the GPUs you want to use (by ID as shown in nvidia-smi)
GPU_IDS=(0 1 2 3 4 5 6 7)

# Base experiment name (will be extended with loss/regularization)
BASE_NAME="state_sm_emb_epoch16_fewshot_v2"

# Search grid
LOSSES=(energy mse se sinkhorn)
REGS=(0 0.1 0.01)

########################################
# Common command pieces
########################################

# Command prefix (env + executable)
CMD_PREFIX=(uv run state tx train)

# Common Hydra/arg overrides (kept as an array to preserve quoting safely)
COMMON_ARGS=(
  'data.kwargs.toml_config_path="competition_support_set/fewshot_v2.toml"'
  'data.kwargs.num_workers=8'
  'data.kwargs.batch_col="batch_var"'
  'data.kwargs.pert_col="target_gene"'
  'data.kwargs.cell_type_key="cell_type"'
  'data.kwargs.control_pert="non-targeting"'
  'data.kwargs.embed_key="X_state"'
  'data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt"'
  'training.max_steps=50000'
  'training.val_freq=2000'
  'training.ckpt_every_n_steps=5000'
  'model=state_sm'
  'wandb.tags=[state_sm_emb_epoch16_fewshot_v2_loss]'
  'output_dir="competition"'
)

########################################
# Helpers
########################################

# Sanitize a floating value for naming (e.g., 0.01 -> 0p01)
sanitize_float() {
  local x="$1"
  echo "${x//./p}"
}

########################################
# Create task list (pairs of loss|reg)
########################################

declare -a TASKS=()
for loss in "${LOSSES[@]}"; do
  for reg in "${REGS[@]}"; do
    TASKS+=("${loss}|${reg}")
  done
done

NUM_GPUS="${#GPU_IDS[@]}"
NUM_TASKS="${#TASKS[@]}"

echo "Running ${NUM_TASKS} tasks across ${NUM_GPUS} GPUs..."
echo

########################################
# Signal handling for graceful shutdown
########################################

cleanup() {
  echo "Interrupt received! Stopping all workers..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Terminating worker PID $pid"
      kill -TERM "$pid" 2>/dev/null
    fi
  done
  
  # Give workers a chance to clean up
  sleep 2
  
  # Force kill any remaining processes
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing worker PID $pid"
      kill -KILL "$pid" 2>/dev/null
    fi
  done
  
  echo "All workers stopped."
  exit 130  # Standard exit code for Ctrl+C
}

# Set up signal trap
trap cleanup SIGINT SIGTERM

########################################
# GPU workers: each GPU runs its slice of tasks
########################################

run_worker() {
  local worker_idx="$1"
  local gpu_id="$2"

  # Handle interrupts in worker
  trap 'echo "[GPU ${gpu_id}] Worker interrupted, stopping..."; exit 130' SIGINT SIGTERM

  for ((i=worker_idx; i<NUM_TASKS; i+=NUM_GPUS)); do
    IFS='|' read -r loss reg <<< "${TASKS[$i]}"

    reg_tag="$(sanitize_float "$reg")"
    name="${BASE_NAME}_loss-${loss}_reg-${reg_tag}"

    echo "[GPU ${gpu_id}] Starting: ${name}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${CMD_PREFIX[@]}" \
      "${COMMON_ARGS[@]}" \
      "model.kwargs.loss=${loss}" \
      "model.kwargs.regularization=${reg}" \
      "name=\"${name}\"" &
    
    # Store the training process PID and wait for it
    train_pid=$!
    wait $train_pid
    train_exit_code=$?
    
    # Check if training was interrupted
    if [ $train_exit_code -ne 0 ]; then
      echo "[GPU ${gpu_id}] Training interrupted or failed: ${name}"
      exit $train_exit_code
    fi

    echo "[GPU ${gpu_id}] Finished: ${name}"
  done
}

# Launch one background worker per GPU
pids=()
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  run_worker "$idx" "$gpu" &
  pids+=("$!")
done

# Wait for all workers with responsive interrupt handling
fail=0
echo "Waiting for all workers to complete (Press Ctrl+C to stop all)..."

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "One or more runs failed or were interrupted."
  exit 1
else
  echo "All runs completed successfully."
fi

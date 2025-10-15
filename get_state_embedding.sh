#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p data_state_emb_epoch_16

# Process all datasets with the specific checkpoint
CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/competition_train.h5 --output data_state_emb_epoch_16/competition_train.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/k562_gwps.h5 --output data_state_emb_epoch_16/k562_gwps.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/rpe1.h5 --output data_state_emb_epoch_16/rpe1.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/jurkat.h5 --output data_state_emb_epoch_16/jurkat.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/k562.h5 --output data_state_emb_epoch_16/k562.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/hepg2.h5 --output data_state_emb_epoch_16/hepg2.h5ad

CUDA_VISIBLE_DEVICES=7 uv run state emb transform --model-folder SE-600M --checkpoint /home/wyunzhe/projects/state/SE-600M/se600m_epoch16.ckpt --input competition_support_set/competition_val_template.h5ad --output data_state_emb_epoch_16/competition_val_template.h5ad
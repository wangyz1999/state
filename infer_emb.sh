CUDA_VISIBLE_DEVICES=0 uv run state tx infer \
  --output "competition/state_sm_emb_2/step=16000.h5ad" \
  --embed_key "X_state" \
  --model_dir "competition/state_sm_emb_2" \
  --checkpoint "competition/state_sm_emb_2/checkpoints/step=16000.ckpt" \
  --adata "data_state_emb/competition_val_template.h5ad" \
  --pert_col "target_gene"
CUDA_VISIBLE_DEVICES=7 uv run state tx infer \
  --output "competition/state_sm_emb/step=step=8800-val_loss=val_loss=1.0023.h5ad" \
  --embed_key "X_state" \
  --model_dir "competition/state_sm_emb" \
  --checkpoint "competition/state_sm_emb/checkpoints/step=step=8800-val_loss=val_loss=1.0023.ckpt" \
  --adata "data_state_emb/competition_val_template.h5ad" \
  --pert_col "target_gene"
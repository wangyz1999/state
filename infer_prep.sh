CUDA_VISIBLE_DEVICES=2 uv run state tx infer \
  --output "competition/state_sm_emb_weight0.1/state_sm_emb_weight0.1_pred.h5ad" \
  --embed_key "X_state" \
  --model_dir "competition/state_sm_emb_weight0.1" \
  --checkpoint "competition/state_sm_emb_weight0.1/checkpoints/step=step=40000-val_loss=val_loss=0.6029.ckpt" \
  --adata "data_state_emb/competition_val_template.h5ad" \
  --pert_col "target_gene"

cell-eval prep -i "competition/state_sm_emb_weight0.1/state_sm_emb_weight0.1_pred.h5ad" -g "competition_support_set/gene_names.csv"
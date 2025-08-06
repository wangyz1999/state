uv run state tx infer \
  --output "competition/prediction_40000_val.h5ad" \
  --model_dir "competition/second_run" \
  --checkpoint "competition/second_run/checkpoints/step=40000.ckpt" \
  --adata "competition_support_set/hepg2.h5" \
  --pert_col "target_gene"
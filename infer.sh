CUDA_VISIBLE_DEVICES=7 uv run state tx infer \
  --output "competition/state_tahoe_freeze_middle/pred_6000.h5ad" \
  --model_dir "competition/state_tahoe_freeze_middle" \
  --checkpoint "competition/state_tahoe_freeze_middle/checkpoints/step=6000.ckpt" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --pert_col "target_gene"
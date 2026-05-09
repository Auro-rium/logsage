# LogSage Training Run Report

## Run Summary

- Run ID: `full-20260509-145106`
- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Method: QLoRA / LoRA adapter fine-tuning
- Instance: AWS EC2 `g5.2xlarge`
- GPU: NVIDIA A10G
- Dataset rows: `1116`
- Train / eval split: `1004 / 112`
- Epochs: `3`
- Max sequence length: `2048`
- Effective batch size: `8`
- Total optimization steps: `378`

## Final Metrics

- Final train loss: `0.7878258203072522`
- Final eval loss: `0.8106855750083923`
- Best eval loss: `0.7893214225769043` at step `250`
- Lowest logged train loss: `0.5278223037719727` at step `370`
- Train runtime: `2058.2405s` (`34.3 min`)
- Train steps/sec: `0.184`
- Eval runtime: `22.2714s`

## Curve Reading

- The loss dropped quickly in the first `100` steps, from about `2.15` to sub-`0.90`.
- Eval loss improved steadily through the middle of training and reached its best point at step `250`.
- After step `250`, eval loss stayed roughly flat in the `0.79` to `0.81` range.
- That shape suggests the run learned the task well enough for this dataset size, and extra epochs beyond `3` were unlikely to buy much.

## Graph

- Metrics plot: [training_metrics.svg](../results/training_metrics.svg)
- Tabular curve export: [training_curves.csv](../results/training_curves.csv)

The graph contains:

- Train loss vs eval loss by step
- Learning rate decay across the run
- Gradient norm by step

## Raw Artifacts

- Metrics stream: [training_metrics.jsonl](../results/training_metrics.jsonl)
- Final metrics summary: [training_summary.json](../results/training_summary.json)
- Eval generations: [training_eval_outputs.jsonl](../results/training_eval_outputs.jsonl)
- Trainer log: [training_train.log](../results/training_train.log)
- TensorBoard event files: [`results/tensorboard/`](../results/tensorboard/)

## Local Backups

- Full adapter backup: `/home/lenovo/LogSage-Qwen2.5-7B-QLoRA-v0`
- Full run backup: `/home/lenovo/full-20260509-145106`
- HF-ready publish folder: `/home/lenovo/LogSage-Qwen2.5-7B-QLoRA-v0-publish`

## Notes

- The adapter and reports were pulled back from EC2 before termination.
- The EC2 instance `i-037646d57f18de5c6` was terminated after the artifacts were saved.
- Hugging Face upload is still blocked by the current token permissions, not by missing model artifacts.

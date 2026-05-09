import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset

from src.data import build_chat_prompt, dataset_stats, format_training_text, load_and_validate_jsonl


DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "LogSage-Qwen2.5-7B-QLoRA-v0"
DEFAULT_HF_REPO_ID = "auro-rirum/LogSage-Qwen2.5-7B-QLoRA-v0"


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def build_metrics_recorder_class(trainer_callback_cls):
    class MetricsRecorder(trainer_callback_cls):
        def __init__(self, metrics_path):
            self.metrics_path = Path(metrics_path)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {"step": state.global_step, **logs}
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return MetricsRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LogSage with QLoRA.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-path", default="data/logs_dataset.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def build_run_paths(args):
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir or Path("runs") / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    return run_id, run_dir


def configure_logging(run_dir):
    log_path = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command_output(command):
    if shutil.which(command[0]) is None:
        return None
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def collect_environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "trl": package_version("trl"),
        "peft": package_version("peft"),
        "unsloth": package_version("unsloth"),
        "nvidia_smi": command_output(["nvidia-smi"]),
    }


def save_normalized_dataset(rows, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_datasets(rows, args, run_dir):
    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=args.eval_size, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    def format_batch(example):
        return {"text": format_training_text(example)}

    train_ds = train_ds.map(format_batch, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_batch, remove_columns=eval_ds.column_names)

    split_stats = {
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "eval_size": args.eval_size,
    }
    write_json(run_dir / "dataset_split.json", split_stats)
    return train_ds, eval_ds


def latest_checkpoint(output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    return str(checkpoints[-1]) if checkpoints else None


def build_model_card(args, run_dir, stats, metrics):
    sample_output = {
        "issue": "Database authentication failed after an SSL policy change.",
        "root_cause": "The application is connecting to PostgreSQL with sslmode=disable while the database now requires SSL.",
        "severity": "high",
        "fix": "Enable SSL in the PostgreSQL connection settings, rotate/retest credentials if needed, and redeploy the service.",
        "confidence": "92%",
    }
    card = f"""---
base_model: {args.model_name}
library_name: peft
tags:
- logs
- qlora
- learning-grade
---

# LogSage QLoRA Adapter

This is a learning-grade QLoRA adapter fine-tuned to analyze application logs and return a structured JSON-style incident summary.

## Base Model

`{args.model_name}`

## Dataset

- Rows: {stats["rows"]}
- Severity distribution: {stats["severity_distribution"]}
- Max input characters: {stats["max_input_chars"]}

## Training

- Epochs: {args.epochs}
- Max sequence length: {args.max_seq_length}
- Train batch size: {args.train_batch_size}
- Gradient accumulation: {args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Warmup steps: {args.warmup_steps}

## Evaluation

Final metrics:

```json
{json.dumps(metrics, indent=2, ensure_ascii=False)}
```

## Intended Use

Use this adapter for experiments and learning around log analysis, incident triage, and instruction fine-tuning workflows. It is not production validated.

## Prompt Format

```text
<|im_start|>system
You are LogSage, a careful incident-analysis assistant. Return only valid JSON with keys: issue, root_cause, severity, fix, confidence.<|im_end|>
<|im_start|>user
Analyze the following logs and identify the issue.

Logs:
<application logs>
<|im_end|>
<|im_start|>assistant
```

## Sample Output

```json
{json.dumps(sample_output, indent=2, ensure_ascii=False)}
```

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "{args.model_name}"
adapter = "{args.hf_repo_id}"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(model, adapter)
```

## Limitations

The dataset is small and synthetic/curated. Outputs should be reviewed by a human before operational use.
"""
    path = run_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run_sample_eval(model, tokenizer, rows, run_dir, max_new_tokens):
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    eval_path = run_dir / "eval_outputs.jsonl"
    sample_rows = rows[: min(5, len(rows))]

    with eval_path.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            prompt = (
                build_chat_prompt(
                    instruction=row["instruction"],
                    logs=row["input"],
                )
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            handle.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "expected": row["output"],
                        "generated": generated,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return eval_path


def push_artifacts_to_hub(args, run_dir, model_card_path):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.hf_repo_id,
        folder_path=args.output_dir,
        path_in_repo=".",
        repo_type="model",
    )
    api.upload_file(
        repo_id=args.hf_repo_id,
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_type="model",
    )
    report_path = run_dir / "training_report.md"
    if report_path.exists():
        api.upload_file(
            repo_id=args.hf_repo_id,
            path_or_fileobj=str(report_path),
            path_in_repo="training_report.md",
            repo_type="model",
        )


def write_training_report(args, run_dir, stats, final_metrics, eval_path, env):
    report = f"""# LogSage Training Report

## Run

- Run directory: `{run_dir}`
- Output directory: `{args.output_dir}`
- Base model: `{args.model_name}`
- CUDA available: `{env["cuda_available"]}`
- GPU: `{env["cuda_device_name"]}`

## Dataset

- Rows: {stats["rows"]}
- Severity distribution: {stats["severity_distribution"]}
- Max input characters: {stats["max_input_chars"]}

## Hyperparameters

- Epochs: {args.epochs}
- Max sequence length: {args.max_seq_length}
- Train batch size: {args.train_batch_size}
- Gradient accumulation steps: {args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}
- Warmup steps: {args.warmup_steps}
- Eval steps: {args.eval_steps}
- Save steps: {args.save_steps}

## Final Metrics

```json
{json.dumps(final_metrics, indent=2, ensure_ascii=False)}
```

## Observability Artifacts

- Console/file log: `{run_dir / "train.log"}`
- TensorBoard: `{run_dir / "tensorboard"}`
- Metrics JSONL: `{run_dir / "metrics" / "metrics.jsonl"}`
- Sample eval generations: `{eval_path}`

## Notes

This is a learning-grade fine-tune. Validate outputs manually before using them for incident decisions.
"""
    path = run_dir / "training_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main():
    args = parse_args()
    run_id, run_dir = build_run_paths(args)
    log_path = configure_logging(run_dir)
    logging.info("Starting LogSage run %s", run_id)
    logging.info("Writing logs to %s", log_path)

    write_json(run_dir / "config.json", vars(args))
    env = collect_environment()
    write_json(run_dir / "environment.json", env)
    logging.info("CUDA available: %s", env["cuda_available"])

    rows = load_and_validate_jsonl(args.data_path)
    stats = dataset_stats(rows)
    write_json(run_dir / "dataset_stats.json", stats)
    save_normalized_dataset(rows, run_dir / "normalized_dataset.jsonl")
    logging.info("Validated %s rows", stats["rows"])

    train_ds, eval_ds = prepare_datasets(rows, args, run_dir)
    logging.info("Train rows: %s; eval rows: %s", len(train_ds), len(eval_ds))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Unsloth training. Run this on the planned EC2 GPU instance."
        )

    # Unsloth must be imported before TRL/Transformers training classes.
    from unsloth import FastLanguageModel
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tensorboard")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_id,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    MetricsRecorder = build_metrics_recorder_class(TrainerCallback)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=[MetricsRecorder(metrics_path)],
    )

    resume_checkpoint = latest_checkpoint(args.output_dir) if args.resume else None
    if resume_checkpoint:
        logging.info("Resuming from checkpoint %s", resume_checkpoint)

    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_metrics = dict(trainer_stats.metrics)
    final_metrics.update(trainer.evaluate())
    write_json(run_dir / "metrics.json", final_metrics)
    logging.info("Final metrics: %s", final_metrics)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Saved adapter and tokenizer to %s", args.output_dir)

    eval_path = run_sample_eval(model, tokenizer, rows, run_dir, args.max_new_tokens)
    report_path = write_training_report(args, run_dir, stats, final_metrics, eval_path, env)
    model_card_path = build_model_card(args, run_dir, stats, final_metrics)
    shutil.copyfile(model_card_path, Path(args.output_dir) / "README.md")
    shutil.copyfile(report_path, Path(args.output_dir) / "training_report.md")

    if args.push_to_hub:
        logging.info("Uploading artifacts to Hugging Face repo %s", args.hf_repo_id)
        push_artifacts_to_hub(args, run_dir, model_card_path)

    logging.info("Done. Run artifacts are in %s", run_dir)


if __name__ == "__main__":
    main()

# LogSage Interview Guide

This guide summarizes the project in the language an interviewer or reviewer is likely to expect.

## One-Minute Pitch

LogSage is a learning-grade log-analysis fine-tuning project. I built a structured JSONL dataset of 1,116 incident examples, validated and normalized the schema, fine-tuned `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` with QLoRA on an AWS `g5.2xlarge` GPU instance, tracked training with metrics and TensorBoard, published the final LoRA adapter to Hugging Face, and added a Colab inference notebook so other people can run it.

The model takes raw application logs and returns a JSON diagnosis with `issue`, `root_cause`, `severity`, `fix`, and `confidence`.

## What Was Actually Trained

This project did not full fine-tune all 7B parameters. The base model stayed mostly frozen, and QLoRA trained small LoRA adapter matrices on top of the quantized base model.

That matters because:

- It makes training practical on a single 24 GB A10G GPU.
- The final adapter is small compared with the full base model.
- Inference requires loading both the base model and the adapter.

## Why QLoRA

QLoRA combines 4-bit quantization with LoRA adapter training. The base weights are loaded in 4-bit form to reduce VRAM use, while trainable low-rank adapter weights learn the task-specific behavior.

For this project, QLoRA was the right fit because:

- Full fine-tuning a 7B model would require much more GPU memory.
- The dataset is small, so full-parameter training would be unnecessary.
- A LoRA adapter is easy to publish and reuse.

## Dataset Design

Each row has:

- `instruction`
- `input`
- `output.issue`
- `output.root_cause`
- `output.severity`
- `output.fix`
- `output.confidence`

Normalization was important because the model learns the target format. Confidence values were normalized to percentage strings, and severity was constrained to `low`, `medium`, or `high`.

## Training Setup

- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Method: QLoRA / PEFT LoRA
- Hardware: AWS EC2 `g5.2xlarge`
- GPU: NVIDIA A10G
- Dataset: 1,116 rows
- Split: 1,004 train / 112 eval
- Epochs: 3
- Steps: 378
- Final train loss: 0.788
- Final eval loss: 0.811
- Best eval loss: 0.789 at step 250
- Runtime: about 34 minutes

## Why Best Eval Loss Was Before The Final Step

The best eval loss happened at step 250, while final training ended at step 378. That is normal. Training loss can keep improving while eval loss flattens or slightly worsens. It suggests the model had mostly learned the dataset pattern by the middle of the third epoch, and extra training did not add much generalization.

## Inference Flow

Inference has five steps:

1. Load the base model.
2. Load the LoRA adapter from Hugging Face.
3. Format the logs using the chat template.
4. Generate tokens.
5. Extract and validate the JSON object.

The JSON extraction step matters because language models can echo the prompt or include extra text. The inference script handles this by extracting the first valid JSON object and validating required fields.

## Why The Hugging Face Repo Needs Colab

The Hugging Face model repo stores the adapter, not a merged standalone model. The normal model-page inference widget expects a complete directly loadable model. For this project, users should run inference in Colab, a local GPU machine, a Hugging Face Space, or a custom endpoint that loads both:

- base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- adapter: `auro-rirum/LogSage-Qwen2.5-7B-QLoRA-v0`

## Strong Answers To Expected Questions

**Why not use a larger dataset?**  
This was a learning-grade project focused on the full fine-tuning workflow end to end. The dataset is enough to demonstrate schema following and log triage behavior, but not enough to claim production reliability.

**Why not merge the adapter into the base model?**  
Keeping the adapter separate is cheaper to store, easier to inspect, and standard for LoRA projects. A merged model can be created later if hosted inference becomes the priority.

**What would you improve next?**  
I would add a held-out benchmark set, compare base model vs adapter outputs, build a small evaluation scorer for JSON validity and field accuracy, and publish a Hugging Face Space for browser-based inference.

**What are the main limitations?**  
The dataset is small, examples are curated, outputs still require human review, and this is not production validated for real incident response.

**What part shows engineering discipline?**  
The project includes dataset validation, normalized targets, checkpointing, eval metrics, TensorBoard artifacts, AWS budget guardrails, local backups, Hugging Face publishing, and reproducible Colab inference.

## Project Takeaway

This is not just a notebook experiment. It is a complete fine-tuning workflow: data preparation, model training, cloud GPU execution, observability, artifact publishing, and documented inference.

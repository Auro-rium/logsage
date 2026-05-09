import argparse

from unsloth import FastLanguageModel

from src.data import build_chat_prompt


DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_DIR = "LogSage-Qwen2.5-7B-QLoRA-v0"


SAMPLE_LOGS = """2026-03-21T18:44:55.004Z ERROR api checkout failed
Error: SASL: SCRAM-SERVER-FINAL-MESSAGE: server signature is missing
    at Object.continueSession (/srv/app/node_modules/pg/lib/crypto/sasl.js:36:11)
    at Client._handleAuthSASLFinal (/srv/app/node_modules/pg/lib/client.js:276:10)
DB_HOST=db.internal DB_PORT=5432 sslmode=disable pool=20
note=RDS instance was switched to require SSL this morning"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run a LogSage adapter inference sample.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--logs", default=SAMPLE_LOGS)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model.load_adapter(args.adapter_dir)
    FastLanguageModel.for_inference(model)

    prompt = build_chat_prompt(
        instruction="Analyze the following logs and identify the issue.",
        logs=args.logs,
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.2,
        do_sample=False,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

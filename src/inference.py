import argparse
import json

from src.data import VALID_SEVERITIES, build_chat_prompt, normalize_confidence


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
    parser.add_argument("--instruction", default="Analyze the following logs and identify the issue.")
    parser.add_argument("--raw-output", action="store_true")
    return parser.parse_args()


def extract_json_object(text):
    assistant_marker = "assistant\n"
    start_idx = text.rfind(assistant_marker)
    search_from = start_idx + len(assistant_marker) if start_idx != -1 else 0

    decoder = json.JSONDecoder()
    for idx in range(search_from, len(text)):
        if text[idx] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in model output")


def validate_inference_output(payload):
    if not isinstance(payload, dict):
        raise ValueError("inference output must be a JSON object")

    required = ("issue", "root_cause", "severity", "fix", "confidence")
    for key in required:
        if key not in payload:
            raise ValueError(f"missing required key {key!r}")
        if key != "confidence" and (
            not isinstance(payload[key], str) or not payload[key].strip()
        ):
            raise ValueError(f"{key!r} must be a non-empty string")

    severity = payload["severity"].strip().lower()
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"severity must be one of {sorted(VALID_SEVERITIES)}")

    return {
        "issue": payload["issue"].strip(),
        "root_cause": payload["root_cause"].strip(),
        "severity": severity,
        "fix": payload["fix"].strip(),
        "confidence": normalize_confidence(payload["confidence"]),
    }


def main():
    args = parse_args()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model.load_adapter(args.adapter_dir)
    FastLanguageModel.for_inference(model)

    prompt = build_chat_prompt(
        instruction=args.instruction,
        logs=args.logs,
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.2,
        do_sample=False,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    structured = validate_inference_output(extract_json_object(decoded))

    if args.raw_output:
        print(decoded)
        print("\n---PARSED-JSON---")

    print(json.dumps(structured, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import json
from collections import Counter
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = ("instruction", "input", "output")
REQUIRED_OUTPUT_KEYS = ("issue", "root_cause", "severity", "fix", "confidence")
VALID_SEVERITIES = {"low", "medium", "high"}


def normalize_confidence(value):
    """Return one stable percentage string for training targets."""
    if isinstance(value, bool):
        raise ValueError("confidence must not be a boolean")
    if isinstance(value, (int, float)):
        numeric = float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            raise ValueError("confidence must not be empty")
        try:
            numeric = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"confidence must be numeric or percentage-like: {value!r}") from exc
    elif not isinstance(value, (int, float)):
        raise ValueError(f"confidence has unsupported type {type(value).__name__}")

    if 0 <= numeric <= 1:
        numeric *= 100
    if not 0 <= numeric <= 100:
        raise ValueError("confidence must be between 0 and 100")
    if numeric.is_integer():
        return f"{int(numeric)}%"
    return f"{numeric:.1f}%"


def normalize_row(row, line_number):
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in row:
            raise ValueError(f"line {line_number}: missing top-level key {key!r}")

    if not isinstance(row["instruction"], str) or not row["instruction"].strip():
        raise ValueError(f"line {line_number}: instruction must be a non-empty string")
    if not isinstance(row["input"], str) or not row["input"].strip():
        raise ValueError(f"line {line_number}: input must be a non-empty string")
    if not isinstance(row["output"], dict):
        raise ValueError(f"line {line_number}: output must be an object")

    output = row["output"]
    for key in REQUIRED_OUTPUT_KEYS:
        if key not in output:
            raise ValueError(f"line {line_number}: missing output key {key!r}")
        if key != "confidence" and (
            not isinstance(output[key], str) or not output[key].strip()
        ):
            raise ValueError(f"line {line_number}: output.{key} must be a non-empty string")

    severity = output["severity"].strip().lower()
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"line {line_number}: output.severity must be one of {sorted(VALID_SEVERITIES)}"
        )

    return {
        "instruction": row["instruction"].strip(),
        "input": row["input"].strip(),
        "output": {
            "issue": output["issue"].strip(),
            "root_cause": output["root_cause"].strip(),
            "severity": severity,
            "fix": output["fix"].strip(),
            "confidence": normalize_confidence(output["confidence"]),
        },
    }


def load_and_validate_jsonl(path):
    rows = []
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            rows.append(normalize_row(row, line_number))

    if not rows:
        raise ValueError(f"{path} contains no training rows")
    return rows


def dataset_stats(rows):
    severities = Counter(row["output"]["severity"] for row in rows)
    max_input_chars = max(len(row["input"]) for row in rows)
    return {
        "rows": len(rows),
        "severity_distribution": dict(sorted(severities.items())),
        "max_input_chars": max_input_chars,
    }


def build_chat_prompt(instruction, logs, response=None):
    user_content = (
        f"{instruction.strip()}\n\n"
        "Logs:\n"
        f"{logs.strip()}"
    )
    prompt = (
        "<|im_start|>system\n"
        "You are LogSage, a careful incident-analysis assistant. "
        "Return only valid JSON with keys: issue, root_cause, severity, fix, confidence."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if response is not None:
        prompt += f"{response}<|im_end|>"
    return prompt


def format_training_text(example):
    output = {
        "issue": example["output"]["issue"],
        "root_cause": example["output"]["root_cause"],
        "severity": example["output"]["severity"],
        "fix": example["output"]["fix"],
        "confidence": normalize_confidence(example["output"]["confidence"]),
    }
    return build_chat_prompt(
        instruction=example["instruction"],
        logs=example["input"],
        response=json.dumps(output, ensure_ascii=False),
    )

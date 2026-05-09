import argparse
import json

from src.data import dataset_stats, load_and_validate_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the LogSage JSONL dataset.")
    parser.add_argument("--data-path", default="data/logs_dataset.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_and_validate_jsonl(args.data_path)
    print(json.dumps(dataset_stats(rows), indent=2))


if __name__ == "__main__":
    main()

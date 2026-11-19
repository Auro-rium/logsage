import unittest
from src.splits import split_rows, dataset_fingerprint
from src.evaluate import evaluate
from src.inference import InferenceOutputError, extract_json_object, validate_inference_output


def row(log, severity="high"):
    return {"input": log, "instruction": "analyze", "output": {"severity": severity}}


class TestCore(unittest.TestCase):
    def test_split_is_reproducible_and_keeps_duplicate_logs_together(self):
        rows = [row(str(i)) for i in range(15)] + [row("  2  ")]
        split = split_rows(rows)
        self.assertEqual(split, split_rows(rows))
        self.assertEqual(sum(any(r["input"].strip() == "2" for r in p) for p in split), 1)
        self.assertEqual(sum(map(len, split)), len(rows))
        self.assertEqual(dataset_fingerprint(rows), dataset_fingerprint(rows))

    def test_evaluation_counts_invalid_as_incorrect(self):
        class Runtime:
            def analyze(self, logs, instruction):
                if logs == "bad":
                    raise InferenceOutputError()
                return {"severity": "high"}
        result = evaluate([row("ok"), row("bad")], Runtime())
        self.assertEqual(result["json_validity_rate"], 0.5)
        self.assertEqual(result["severity_accuracy"], 0.5)

    def test_output_validation(self):
        payload = extract_json_object('prefix {"issue":"x","root_cause":"y","severity":"HIGH","fix":"z","confidence":"90%"}')
        self.assertEqual(validate_inference_output(payload)["severity"], "high")
        with self.assertRaises(ValueError):
            validate_inference_output({})


if __name__ == "__main__":
    unittest.main()

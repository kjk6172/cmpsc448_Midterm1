"""Meaningful tests for leakage prevention, input semantics, and padding masks."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_attribution.data import (Row, Vocabulary, collate_batch, select_rows,
                                 split_prompts, tokenize, validate_rows, verify_split)
from llm_attribution.experiment import TrainingConfig, run_neural
from llm_attribution.models import BiLSTM, TextCNN
from llm_attribution.predict import predict


def fixture_rows(n_prompts=30):
    rows = []
    for i in range(n_prompts):
        for label in ("Claude", "GPT", "Gemini"):
            rows.append(Row(str(i), label, f"Question {i} describes shared task", f"{label} answer to question {i}",
                            "writing" if i % 2 else "general", "source_a" if i % 2 else "source_b",
                            "template_shared" if i in (0, 1) else f"group_{i}"))
    return rows


class DataTests(unittest.TestCase):
    def test_all_model_answers_and_templates_remain_in_same_partition(self):
        rows = fixture_rows()
        split = split_prompts(rows)
        verify_split(rows, split)
        partitions = {pid: name for name, ids in split.items() for pid in ids}
        self.assertEqual(partitions["0"], partitions["1"])
        for name, ids in split.items():
            selected = select_rows(rows, ids)
            self.assertEqual(len(selected), 3 * len(ids))
        self.assertEqual(split, split_prompts(rows))

    def test_group_overlap_is_rejected(self):
        rows = fixture_rows()
        split = split_prompts(rows)
        for name in split:
            split[name] = [pid for pid in split[name] if pid not in ("0", "1")]
        split["train"].append("0")
        split["test"].append("1")
        with self.assertRaisesRegex(ValueError, "template group"):
            verify_split(rows, split)

    def test_source_holdout_purges_shared_non_target_prompt(self):
        rows = fixture_rows()
        split = split_prompts(rows, holdout_source="source_a")
        self.assertIn("0", split["excluded"])
        self.assertIn("1", split["test"])
        self.assertNotIn("0", split["train"] + split["validation"])
        self.assertTrue(all(row.source_dataset == "source_a" for row in select_rows(rows, split["test"])))

    def test_vocabulary_does_not_learn_test_words(self):
        vocab = Vocabulary.fit(["trainword trainword"], min_frequency=1)
        self.assertEqual(vocab.encode("testonlyword", 10), [1])
        self.assertNotIn("testonlyword", vocab.stoi)

    def test_literal_separator_word_is_not_reserved_token(self):
        self.assertEqual(tokenize("a separator [SEP] b"), ["a", "separator", "[SEP]", "b"])

    def test_long_question_retains_same_answer_budget_as_output_only(self):
        row = Row("1", "GPT", "question " * 300, "answer " * 300, "general")
        vocab = Vocabulary.fit([row.text("input_output")], min_frequency=1)
        combined = vocab.encode_row(row, "input_output", output_budget=256, input_budget=64)
        output = vocab.encode_row(row, "output", output_budget=256, input_budget=64)
        self.assertEqual(len(combined), 321)
        self.assertEqual(combined[64], vocab.stoi["[SEP]"])
        self.assertEqual(combined[65:], output)

    def test_missing_family_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_rows(fixture_rows()[:-1])

    def test_identical_question_different_id_is_rejected(self):
        rows = fixture_rows()
        rows += [Row("duplicate", row.llm_name, row.llm_input, row.llm_output, row.domain)
                 for row in rows[:3]]
        with self.assertRaisesRegex(ValueError, "Identical question"):
            validate_rows(rows)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_cnn_predictions_are_invariant_to_added_padding(self):
        model = TextCNN(20, 3, embedding_dim=8, hidden_dim=4, dropout=0).eval()
        tokens = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 0, 0, 0]])
        lengths = torch.tensor([5, 2])
        extra_padding = torch.nn.functional.pad(tokens, (0, 9))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(tokens, lengths), model(extra_padding, lengths), atol=1e-6))

    def test_lstm_predictions_are_invariant_to_added_padding(self):
        model = BiLSTM(20, 3, embedding_dim=8, hidden_dim=4, dropout=0).eval()
        tokens = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 0, 0, 0]])
        lengths = torch.tensor([5, 2])
        extra_padding = torch.nn.functional.pad(tokens, (0, 9))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(tokens, lengths), model(extra_padding, lengths), atol=1e-6))

    def test_identical_questions_yield_exactly_chance_accuracy(self):
        vocab = Vocabulary.fit(["same question repeated"], min_frequency=1)
        encoded = vocab.encode("same question repeated", 64)
        tokens, lengths, _ = collate_batch([(encoded, target) for target in range(3)])
        for model in (TextCNN(len(vocab.tokens), 3), BiLSTM(len(vocab.tokens), 3)):
            model.eval()
            with torch.no_grad():
                predictions = model(tokens, lengths).argmax(dim=1)
            self.assertEqual(float((predictions == torch.arange(3)).sum()), 1.0)

    def test_tiny_training_checkpoint_and_prediction_are_valid(self):
        rows = fixture_rows(12)
        split = split_prompts(rows)
        with tempfile.TemporaryDirectory() as directory:
            result = run_neural(rows, split, Path(directory), "cnn", "output", 42,
                                TrainingConfig(epochs=2, patience=2, embedding_dim=8, hidden_dim=4,
                                               batch_size=8, max_length=16, min_frequency=1, threads=1), "fixturehash")
            run_dir = Path(directory) / "runs" / result["run_id"]
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            response = predict(run_dir / "checkpoint.pt", "shared question", "some answer")
            self.assertAlmostEqual(sum(response["probabilities"].values()), 1.0, places=6)
            self.assertEqual(result["n_test_rows"], 3 * len(split["test"]))
            self.assertGreaterEqual(result["accuracy"], 0.0)
            self.assertLessEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

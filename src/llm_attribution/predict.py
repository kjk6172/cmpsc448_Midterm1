"""Inference from a saved seed42 neural checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import Row, Vocabulary, collate_batch
from .models import make_model


def predict(checkpoint: Path, question: str, answer: str) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    vocab = Vocabulary(json.loads(checkpoint.with_name("vocab.json").read_text(encoding="utf-8")))
    model = make_model(state["model_name"], **state["model_kwargs"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    row = Row("inference", "unknown", question, answer, "unknown")
    encoded = vocab.encode_row(row, state["mode"], state["max_length"], state["input_budget"], state["normalize"])
    tokens, lengths, _ = collate_batch([(encoded, 0)])
    with torch.no_grad():
        probabilities = torch.softmax(model(tokens, lengths), dim=1)[0].tolist()
    return {"prediction": state["labels"][max(range(len(probabilities)), key=lambda i: probabilities[i])],
            "probabilities": dict(zip(state["labels"], probabilities)), "run_id": state["run_id"],
            "scope": "Closed-set prediction among this dataset's three source families; probabilities are uncalibrated."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--question", default="")
    parser.add_argument("--answer", default="")
    parser.add_argument("--answer-file", type=Path)
    args = parser.parse_args()
    answer = args.answer_file.read_text(encoding="utf-8") if args.answer_file else args.answer
    print(json.dumps(predict(args.checkpoint, args.question, answer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

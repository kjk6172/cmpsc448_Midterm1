"""Check submission completeness and consistency against saved predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pypdf import PdfReader
import pymupdf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, require_pdf: bool = True) -> dict:
    sys.path.insert(0, str(root / "src"))
    from llm_attribution.experiment import implementation_fingerprint
    from llm_attribution.predict import predict
    from llm_attribution.data import load_rows, select_rows, Vocabulary, TextDataset, collate_batch
    from llm_attribution.models import make_model
    from llm_attribution.experiment import evaluate_neural
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    data_path = root / "data/processed/dataset.csv"
    with data_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {row["prompt_id"]: row["split_group_id"] for row in rows}
    source_rows = {(row["prompt_id"], row["llm_name"]): row for row in rows}
    structured_rows = load_rows(data_path)
    labels = sorted({row["llm_name"] for row in rows})
    if len(labels) < 3:
        raise ValueError("Fewer than three LLM families")
    audit = read_json(root / "data/processed/audit.json")
    forbidden_sources = {"koala", "oasst"}
    if any(row["source_dataset"] in forbidden_sources for row in rows):
        raise ValueError("Excluded Koala/OASST source reintroduced into the submission data")
    if set(audit.get("excluded_sources", [])) != forbidden_sources:
        raise ValueError("Curation audit does not record the required source exclusions")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != audit["processed_file_sha256"]["dataset.csv"]:
        raise ValueError("Curated data differs from its audit checksum")
    jsonl_path = root / "data/processed/dataset.jsonl"
    if hashlib.sha256(jsonl_path.read_bytes()).hexdigest() != audit["processed_file_sha256"]["dataset.jsonl"]:
        raise ValueError("Curated JSONL differs from its audit checksum")
    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    if len(jsonl_rows) != len(rows) or any(
        any(str(record[field]) != row[field] for field in row)
        or record["LLM_name"] != row["llm_name"]
        or record["LLM_Input"] != row["llm_input"]
        or record["LLM_output"] != row["llm_output"]
        for record, row in zip(jsonl_rows, rows)
    ):
        raise ValueError("Curated CSV and JSONL do not describe the same responses")
    results = read_json(root / "artifacts/results.json")
    split = read_json(root / "artifacts/split.json")
    parts = {name: set(split[name]) for name in ("train", "validation", "test")}
    clustered = {name: {groups[pid] for pid in ids} for name, ids in parts.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if parts[left] & parts[right] or clustered[left] & clustered[right]:
            raise ValueError(f"Leakage between {left} and {right}")
    if set().union(*parts.values()) != set(groups):
        raise ValueError("Main split does not cover all prompts")
    configurations = set()
    checkpoints_checked = 0
    checkpoint_predictions_checked = 0
    expected_seeds = set(results["metadata"]["training_seeds"])
    for run in results["runs"]:
        if run["dataset_sha256"] != digest:
            raise ValueError(f"Run used different data: {run['run_id']}")
        run_dir = root / "artifacts/runs" / run["run_id"]
        with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        keys = [(row["prompt_id"], row["actual_label"]) for row in predictions]
        expected_keys = {(pid, label) for pid in run["test_prompt_ids"] for label in labels}
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError(f"Duplicated or incomplete prediction keys: {run['run_id']}")
        target = [labels.index(row["actual_label"]) for row in predictions]
        predicted = [labels.index(row["predicted_label"]) for row in predictions]
        if len(predictions) != run["n_test_rows"]:
            raise ValueError(f"Wrong prediction count: {run['run_id']}")
        if not np.isclose(accuracy_score(target, predicted), run["accuracy"], atol=1e-12):
            raise ValueError(f"Accuracy differs from saved predictions: {run['run_id']}")
        actual_f1 = f1_score(target, predicted, labels=range(len(labels)), average="macro", zero_division=0)
        if not np.isclose(actual_f1, run["macro_f1"], atol=1e-12):
            raise ValueError(f"F1 differs from saved predictions: {run['run_id']}")
        if confusion_matrix(target, predicted, labels=range(len(labels))).tolist() != run["confusion_matrix"]:
            raise ValueError(f"Confusion matrix differs: {run['run_id']}")
        for row in predictions:
            probabilities = np.array([float(row[f"p_{label}"]) for label in labels])
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
                raise ValueError(f"Invalid probabilities: {run['run_id']}")
            if labels[int(probabilities.argmax())] != row["predicted_label"]:
                raise ValueError(f"Probability/prediction inconsistency: {run['run_id']}")
        if run["experiment"] == "main":
            if {row["prompt_id"] for row in predictions} != parts["test"]:
                raise ValueError(f"Run used a different main test set: {run['run_id']}")
            if run["model"] in ("cnn", "lstm"):
                configurations.add((run["model"], run["mode"], run["seed"]))
                if run["mode"] == "input" and not np.isclose(run["accuracy"], 1 / len(labels)):
                    raise ValueError(f"Matched input-only control differs from chance: {run['run_id']}")
                if run["seed"] == 42 and not all((run_dir / name).is_file() for name in ("checkpoint.pt", "vocab.json")):
                    raise ValueError(f"Missing inference checkpoint: {run['run_id']}")
        if run["model"] in ("cnn", "lstm") and run["seed"] == 42:
            checkpoint_path = run_dir / "checkpoint.pt"
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            for name in ("run_id", "dataset_sha256", "split_sha256", "implementation_sha256", "mode", "labels"):
                if state[name] != run[name]:
                    raise ValueError(f"Checkpoint identity differs: {run['run_id']} / {name}")
            if state["model_name"] != run["model"] or state["implementation_sha256"] != implementation_fingerprint():
                raise ValueError(f"Checkpoint implementation differs: {run['run_id']}")
            vocab = read_json(run_dir / "vocab.json")
            if len(vocab) != state["model_kwargs"]["vocab_size"]:
                raise ValueError(f"Checkpoint vocabulary size differs: {run['run_id']}")
            example = predictions[0]
            source = source_rows[(example["prompt_id"], example["actual_label"])]
            replay = predict(checkpoint_path, source["llm_input"], source["llm_output"])
            if replay["prediction"] != example["predicted_label"] or not all(
                np.isclose(replay["probabilities"][label], float(example[f"p_{label}"]), atol=1e-6)
                for label in labels
            ):
                raise ValueError(f"Checkpoint replay differs from saved predictions: {run['run_id']}")
            checkpoints_checked += 1
            model = make_model(state["model_name"], **state["model_kwargs"])
            model.load_state_dict(state["state_dict"])
            replay_rows = select_rows(structured_rows, run["test_prompt_ids"])
            dataset = TextDataset(replay_rows, state["mode"], Vocabulary(vocab),
                                  {label: i for i, label in enumerate(labels)},
                                  state["max_length"], state["normalize"], state["input_budget"])
            _, replay_probabilities = evaluate_neural(
                model, DataLoader(dataset, batch_size=32, collate_fn=collate_batch), labels)
            saved_probabilities = np.array([[float(row[f"p_{label}"]) for label in labels]
                                           for row in predictions])
            replay_keys = [(row.prompt_id, row.llm_name) for row in replay_rows]
            if replay_keys != keys or not np.allclose(replay_probabilities, saved_probabilities, atol=1e-6):
                raise ValueError(f"Full checkpoint replay differs: {run['run_id']}")
            checkpoint_predictions_checked += len(replay_rows)
    expected = {(model, mode, seed) for model in ("cnn", "lstm")
                for mode in ("input", "output", "input_output") for seed in expected_seeds}
    if configurations != expected:
        raise ValueError(f"Missing required neural configurations: {expected - configurations}")
    for experiment in ("domain_holdout", "ablation"):
        if not any(run["experiment"] == experiment and run["model"] in ("cnn", "lstm") for run in results["runs"]):
            raise ValueError(f"Missing extra-credit experiment: {experiment}")
    statistics = read_json(root / "artifacts/statistical_analysis.json")
    if len(statistics["comparisons"]) < 7:
        raise ValueError("Incomplete paired uncertainty analysis")
    for name in ("README.md", "metadata.json", "requirements.txt", "requirements-lock.txt",
                 "data/source_manifest.json", "data/NOTICE.md", "docs/DEFENSE_GUIDE_KO.md"):
        if not (root / name).is_file():
            raise ValueError(f"Missing submission file: {name}")
    pdf_pages = None
    pdf_images_checked = None
    if require_pdf:
        reader = PdfReader(root / "report/project_report.pdf")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = read_json(root / "metadata.json")
        for name in metadata["authors"]:
            if name not in text:
                raise ValueError(f"Author absent from PDF: {name}")
        for rq in ("RQ1", "RQ2", "RQ3", "RQ4"):
            if rq not in text:
                raise ValueError(f"Research question absent from PDF: {rq}")
        for lesson in ("Source review changed", "A low control score", "equal response budget",
                       "Observed differences need uncertainty", "checking artifacts"):
            if lesson not in text:
                raise ValueError(f"Concrete project lesson absent from PDF: {lesson}")
        for placeholder in ("TODO", "TBD", "PLACEHOLDER", "INSERT RESULTS"):
            if placeholder in text:
                raise ValueError(f"Unfinished report text: {placeholder}")
        pdf_pages = len(reader.pages)
        visual_document = pymupdf.open(root / "report/project_report.pdf")
        pdf_images_checked = 0
        for page_number, page in enumerate(visual_document, start=1):
            for info in page.get_image_info():
                bounds = pymupdf.Rect(info["bbox"])
                if not page.rect.contains(bounds):
                    raise ValueError(f"PDF figure clipped outside page {page_number}: {bounds}")
                pdf_images_checked += 1
        if pdf_images_checked < 6:
            raise ValueError("Final report is missing expected result figures")
        visual_document.close()
    return {
        "status": "passed", "dataset_sha256": digest, "matched_prompts": len(groups),
        "near_duplicate_groups": len(set(groups.values())), "responses": len(rows),
        "runs_checked": len(results["runs"]), "required_neural_runs": len(configurations),
        "checkpoints_replayed": checkpoints_checked,
        "checkpoint_predictions_replayed": checkpoint_predictions_checked,
        "split_prompts": {name: len(ids) for name, ids in parts.items()},
        "split_groups": {name: len(ids) for name, ids in clustered.items()},
        "pdf_pages": pdf_pages,
        "pdf_images_checked": pdf_images_checked,
        "checks": ["CSV and JSONL checksums and equivalence", "Koala and OASST absent from all splits", "prompt and template isolation", "required experiment coverage",
                   "recomputed metrics and confusion matrices", "prediction probabilities",
                   "input-only chance control", "inference artifacts", "extra-credit analyses"]
                  + (["author names and research questions in PDF", "no unfinished report placeholders", "PDF figures inside page bounds"]
                     if require_pdf else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    result = verify(args.root, require_pdf=not args.skip_pdf)
    output = args.root / "artifacts/verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

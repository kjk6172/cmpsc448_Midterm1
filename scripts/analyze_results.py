"""Paired bootstrap of saved predictions, clustering near-duplicate prompts.

This script never trains models or changes their selection. It audits saved
metrics and quantifies test-sample uncertainty for pre-specified comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]


def load_predictions(artifacts: Path, run: dict) -> list[dict]:
    with (artifacts / "runs" / run["run_id"] / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))
    return sorted(records, key=lambda row: (row["prompt_id"], row["actual_label"]))


def scores(confusions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1)
    totals = confusions.sum(axis=(-2, -1))
    accuracy = true_positive.sum(axis=-1) / totals
    denominator = confusions.sum(axis=-1) + confusions.sum(axis=-2)
    f1 = np.divide(2 * true_positive, denominator,
                   out=np.zeros_like(true_positive, dtype=float), where=denominator > 0)
    return accuracy, f1.mean(axis=-1)


def comparison(artifacts: Path, name: str, left: list[dict], right: list[dict],
               prompt_groups: dict[str, str], repetitions: int, seed: int) -> dict:
    left = sorted(left, key=lambda run: run["seed"])
    right = sorted(right, key=lambda run: run["seed"])
    if not left or [run["seed"] for run in left] != [run["seed"] for run in right]:
        raise ValueError(f"Comparison {name} has unmatched training seeds")
    labels = left[0]["labels"]
    label_ids = {label: index for index, label in enumerate(labels)}
    exemplar = load_predictions(artifacts, left[0])
    keys = [(row["prompt_id"], row["actual_label"]) for row in exemplar]
    groups = sorted({prompt_groups[row["prompt_id"]] for row in exemplar})
    group_ids = {group: index for index, group in enumerate(groups)}
    cluster_index = [group_ids[prompt_groups[row["prompt_id"]]] for row in exemplar]
    target = [label_ids[row["actual_label"]] for row in exemplar]
    cluster_confusions = []
    for side in (left, right):
        seed_confusions = []
        for run in side:
            records = load_predictions(artifacts, run)
            if [(row["prompt_id"], row["actual_label"]) for row in records] != keys:
                raise ValueError(f"Unpaired test rows for {name}: {run['run_id']}")
            cm = np.zeros((len(groups), len(labels), len(labels)), dtype=float)
            predicted = [label_ids[row["predicted_label"]] for row in records]
            np.add.at(cm, (cluster_index, target, predicted), 1)
            acc, macro_f1 = scores(cm.sum(axis=0))
            if not np.isclose(acc, run["accuracy"], atol=1e-12):
                raise ValueError(f"Saved accuracy does not match predictions: {run['run_id']}")
            if not np.isclose(macro_f1, run["macro_f1"], atol=1e-12):
                raise ValueError(f"Saved Macro-F1 does not match predictions: {run['run_id']}")
            seed_confusions.append(cm)
        cluster_confusions.append(np.asarray(seed_confusions))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    counts = np.zeros((repetitions, len(groups)), dtype=float)
    np.add.at(counts, (np.repeat(np.arange(repetitions), len(groups)), sampled.ravel()), 1)
    side_bootstrap = []
    side_observed = []
    for cms in cluster_confusions:
        observed = scores(cms.sum(axis=1))
        side_observed.append([float(metric.mean()) for metric in observed])
        boot = scores(np.einsum("bg,sgij->sbij", counts, cms))
        side_bootstrap.append([metric.mean(axis=0) for metric in boot])
    metrics = {}
    for index, metric_name in enumerate(("accuracy", "macro_f1")):
        delta = side_bootstrap[0][index] - side_bootstrap[1][index]
        metrics[metric_name] = {
            "left": side_observed[0][index], "right": side_observed[1][index],
            "delta": side_observed[0][index] - side_observed[1][index],
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    return {
        "comparison": name, "left_run_ids": [run["run_id"] for run in left],
        "right_run_ids": [run["run_id"] for run in right],
        "training_seeds": [run["seed"] for run in left], "test_rows": len(keys),
        "test_prompts": len({row["prompt_id"] for row in exemplar}),
        "test_clusters": len(groups), "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/dataset.csv")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with args.data.open(encoding="utf-8-sig", newline="") as handle:
        prompt_groups = {row["prompt_id"]: row.get("split_group_id") or row["prompt_id"]
                         for row in csv.DictReader(handle)}
    runs = json.loads((args.artifacts / "results.json").read_text(encoding="utf-8"))["runs"]
    main_runs = [run for run in runs if run["experiment"] == "main" and run["variant"] == "raw"]
    def selected(model: str, mode: str) -> list[dict]:
        return [run for run in main_runs if run["model"] == model and run["mode"] == mode]
    comparisons = []
    for model in ("cnn", "lstm"):
        comparisons.append(comparison(args.artifacts, f"{model}_input_output_minus_output",
            selected(model, "input_output"), selected(model, "output"),
            prompt_groups, args.repetitions, args.seed))
    comparisons.append(comparison(args.artifacts, "cnn_minus_lstm_output",
        selected("cnn", "output"), selected("lstm", "output"),
        prompt_groups, args.repetitions, args.seed))
    for model in ("cnn", "lstm"):
        raw = [run for run in selected(model, "output") if run["seed"] == 42]
        for variant in ("normalized", "length64"):
            variant_runs = [run for run in runs if run["experiment"] == "ablation"
                            and run["model"] == model and run["variant"] == variant]
            if variant_runs:
                comparisons.append(comparison(args.artifacts, f"{model}_{variant}_minus_raw",
                    variant_runs, raw, prompt_groups, args.repetitions, args.seed))
    lexical_signals = {}
    lexical_run = next((run for run in main_runs if run["model"] == "tfidf"
                        and run["mode"] == "output"), None)
    if lexical_run:
        classifier = joblib.load(args.artifacts / "runs" / lexical_run["run_id"] / "model.joblib")
        terms = classifier.named_steps["tfidf"].get_feature_names_out()
        weights = classifier.named_steps["classifier"].coef_
        for index, label in enumerate(lexical_run["labels"]):
            top_indices = np.argsort(weights[index])[-8:][::-1]
            lexical_signals[label] = [{"ngram": str(terms[i]), "coefficient": float(weights[index, i])}
                                      for i in top_indices]
    result = {
        "method": "Paired percentile bootstrap, resampling complete near-duplicate prompt components; average metric over the same training seeds before differencing.",
        "cluster_key": "split_group_id", "repetitions": args.repetitions, "seed": args.seed,
        "interval": "95% percentile", "comparisons": comparisons,
        "lexical_signals": lexical_signals,
        "lexical_signals_method": "Largest positive coefficients in the training-only response TF-IDF logistic classifier. These are associations in the baseline, not explanations of CNN/LSTM decisions or intrinsic provider traits.",
        "limitations": [
            "Intervals describe uncertainty from this fixed test sample; they do not represent new providers or versions.",
            "Training-seed variability is reported separately; bootstrap does not resample seeds.",
            "Exploratory intervals are not adjusted for multiple comparisons; no confirmatory significance claim is made.",
        ],
    }
    destination = args.artifacts / "statistical_analysis.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(comparisons)} paired comparisons to {destination}")


if __name__ == "__main__":
    main()

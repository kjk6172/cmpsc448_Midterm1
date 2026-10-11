"""Training, validation-only model selection, baselines, and audit artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from .data import (FEATURE_NAMES, Row, TextDataset, Vocabulary, collate_batch,
                   handcrafted_features, select_rows)
from .metrics import classification_metrics, cluster_bootstrap_accuracy
from .models import make_model


@dataclass
class TrainingConfig:
    embedding_dim: int = 64
    hidden_dim: int = 48
    dropout: float = 0.3
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    batch_size: int = 32
    max_length: int = 256
    input_budget: int = 64
    max_vocab: int = 20000
    min_frequency: int = 2
    epochs: int = 20
    patience: int = 4
    clip_gradient_norm: float = 1.0
    threads: int = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(records[0]) if records else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def environment_info() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "device": "cpu", "cpu_count": os.cpu_count()}


def split_fingerprint(split: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps({name: sorted(ids) for name, ids in split.items()},
                                    sort_keys=True).encode()).hexdigest()


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    files = sorted(package_root.glob("*.py")) + [package_root.parents[1] / "scripts" / "run_experiments.py"]
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_neural(model: nn.Module, loader: DataLoader, labels: list[str]) -> tuple[dict, np.ndarray]:
    model.eval()
    probabilities, actual = [], []
    total_loss = 0.0
    with torch.no_grad():
        for tokens, lengths, targets in loader:
            logits = model(tokens, lengths)
            total_loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            probabilities.append(torch.softmax(logits, dim=1).numpy())
            actual.extend(targets.tolist())
    probs = np.concatenate(probabilities)
    metrics = classification_metrics(actual, probs.argmax(axis=1), labels)
    metrics["loss"] = total_loss / len(actual)
    return metrics, probs


def prediction_records(rows: list[Row], probabilities: np.ndarray, labels: list[str]) -> list[dict]:
    predicted = probabilities.argmax(axis=1)
    return [{"prompt_id": row.prompt_id, "split_group_id": row.group_id, "domain": row.domain,
             "source_dataset": row.source_dataset,
             "actual_label": row.llm_name, "predicted_label": labels[int(predicted[i])],
             **{f"p_{label}": float(probabilities[i, j]) for j, label in enumerate(labels)}}
            for i, row in enumerate(rows)]


def run_neural(rows: list[Row], split: dict[str, list[str]], out: Path, model_name: str,
               mode: str, seed: int, config: TrainingConfig, dataset_sha256: str,
               experiment: str = "main", variant: str = "raw", heldout_domain: str | None = None,
               force: bool = False, heldout_source: str | None = None) -> dict:
    target = heldout_source or heldout_domain
    suffix = f"_{target}" if target else ""
    run_id = f"{experiment}{suffix}_{model_name}_{mode}_{variant}_seed{seed}"
    run_dir = out / "runs" / run_id
    config_dict = asdict(config)
    identity = {"run_id": run_id, "model": model_name, "mode": mode, "seed": seed,
                "experiment": experiment, "variant": variant, "heldout_domain": heldout_domain,
                "heldout_source": heldout_source,
                "dataset_sha256": dataset_sha256, "split_sha256": split_fingerprint(split),
                "implementation_sha256": implementation_fingerprint(), "encoding_revision": "equal-output-budget-v1",
                "config": config_dict}
    if not force and (run_dir / "metrics.json").exists():
        saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in identity.items()):
            print(f"RESUME {run_id}: already completed", flush=True)
            return saved
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    labels = sorted({row.llm_name for row in rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    parts = {name: select_rows(rows, ids) for name, ids in split.items()}
    normalize = variant == "normalized"
    max_length = 64 if variant == "length64" else config.max_length
    vocab = Vocabulary.fit((row.text(mode, normalize) for row in parts["train"]),
                           config.max_vocab, config.min_frequency)
    datasets = {name: TextDataset(part, mode, vocab, label_to_id, max_length, normalize, config.input_budget)
                for name, part in parts.items()}
    rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=0, generator=rng)
    # Test data are not evaluated until the selected checkpoint is restored.
    val_loader = DataLoader(datasets["validation"], batch_size=config.batch_size,
                            collate_fn=collate_batch, num_workers=0)
    model_kwargs = {"vocab_size": len(vocab.tokens), "num_classes": len(labels),
                    "embedding_dim": config.embedding_dim, "hidden_dim": config.hidden_dim,
                    "dropout": config.dropout}
    model = make_model(model_name, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    effective_max_length = config.input_budget if mode == "input" else (
        max_length + config.input_budget + 1 if mode == "input_output" else max_length)
    run_config = {**identity, "labels": labels, "effective_max_length": effective_max_length,
                  "output_token_budget": max_length, "input_token_budget": config.input_budget,
                  "vocab_size": len(vocab.tokens), "model_kwargs": model_kwargs,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "environment": environment_info(),
                  "selection_rule": "highest validation macro-F1, then lowest validation loss; no test-based selection"}
    write_json(run_dir / "run_config.json", run_config)
    history = []
    best_f1, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    best_state = None
    print(f"START {run_id} train={len(parts['train'])} val={len(parts['validation'])} test={len(parts['test'])}", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for tokens, lengths, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradient_norm)
            optimizer.step()
            train_loss += float(loss.detach()) * len(targets)
        validation, _ = evaluate_neural(model, val_loader, labels)
        history.append({"epoch": epoch, "train_loss": train_loss / len(datasets["train"]),
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                        "validation_macro_f1": validation["macro_f1"]})
        write_csv(run_dir / "history.csv", history)
        improved = (validation["macro_f1"] > best_f1 + 1e-10 or
                    (abs(validation["macro_f1"] - best_f1) <= 1e-10 and validation["loss"] < best_loss - 1e-8))
        if improved:
            best_f1, best_loss, best_epoch, stale = validation["macro_f1"], validation["loss"], epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print(f"EPOCH {run_id} {epoch}/{config.epochs} train_loss={history[-1]['train_loss']:.4f} "
              f"val_f1={validation['macro_f1']:.4f} stale={stale}", flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size,
                             collate_fn=collate_batch, num_workers=0)
    metrics, probabilities = evaluate_neural(model, test_loader, labels)
    actual = [label_to_id[row.llm_name] for row in parts["test"]]
    metrics["bootstrap_accuracy_ci95"] = cluster_bootstrap_accuracy(actual, probabilities.argmax(axis=1),
                                                                   [row.group_id for row in parts["test"]])
    result = {**run_config, **metrics, "train_rows": len(parts["train"]),
              "validation_rows": len(parts["validation"]), "best_epoch": best_epoch,
              "epochs_run": len(history), "best_validation_macro_f1": best_f1,
              "seconds": time.perf_counter() - started, "labels": labels,
              "split_group_counts": {name: len({row.group_id for row in part}) for name, part in parts.items()},
              "excluded_rows": len(parts.get("excluded", [])),
              "test_prompt_ids": split["test"]}
    write_csv(run_dir / "predictions.csv", prediction_records(parts["test"], probabilities, labels))
    if seed == 42:
        vocab.save(run_dir / "vocab.json")
        torch.save({"state_dict": model.state_dict(), "model_name": model_name,
                    "model_kwargs": model_kwargs, "labels": labels, "mode": mode,
                    "max_length": max_length, "input_budget": config.input_budget, "normalize": normalize,
                    "split_sha256": identity["split_sha256"], "implementation_sha256": identity["implementation_sha256"],
                    "dataset_sha256": dataset_sha256, "run_id": run_id}, run_dir / "checkpoint.pt")
    write_json(run_dir / "metrics.json", result)
    print(f"DONE {run_id} test_accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"seconds={result['seconds']:.1f}", flush=True)
    return result


def run_baseline(rows: list[Row], split: dict[str, list[str]], out: Path, model_name: str,
                 mode: str, seed: int, dataset_sha256: str, experiment: str = "main",
                 heldout_domain: str | None = None, heldout_source: str | None = None) -> dict:
    import joblib
    started = time.perf_counter()
    labels = sorted({row.llm_name for row in rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    parts = {name: select_rows(rows, ids) for name, ids in split.items()}
    texts = {name: [row.text(mode) for row in part] for name, part in parts.items()}
    targets = {name: np.asarray([label_to_id[row.llm_name] for row in part]) for name, part in parts.items()}
    target = heldout_source or heldout_domain
    suffix = f"_{target}" if target else ""
    run_id = f"{experiment}{suffix}_{model_name}_{mode}_raw_seed{seed}"
    run_dir = out / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_c, best_val = None, -1.0
    if model_name in ("tfidf", "features"):
        best_model = None
        feature_parts = {name: handcrafted_features(text) for name, text in texts.items()} if model_name == "features" else texts
        for regularization in (0.1, 1.0, 10.0):
            if model_name == "tfidf":
                classifier = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000,
                                               sublinear_tf=True, lowercase=True)),
                    ("classifier", LogisticRegression(C=regularization, max_iter=1500, random_state=seed))])
            else:
                classifier = Pipeline([("standardize", StandardScaler()),
                                       ("classifier", LogisticRegression(C=regularization, max_iter=1500,
                                                                         random_state=seed))])
            classifier.fit(feature_parts["train"], targets["train"])
            predicted = classifier.predict(feature_parts["validation"])
            val_metrics = classification_metrics(targets["validation"], predicted, labels)
            if val_metrics["macro_f1"] > best_val:
                best_val, best_c, best_model = val_metrics["macro_f1"], regularization, classifier
        probabilities = best_model.predict_proba(feature_parts["test"])
        if not np.array_equal(best_model.classes_, np.arange(len(labels))):
            raise ValueError("Baseline probability columns do not match label order")
        joblib.dump(best_model, run_dir / "model.joblib")
        if model_name == "features":
            coefficients = best_model.named_steps["classifier"].coef_
            write_csv(run_dir / "feature_coefficients.csv",
                      [{"feature": feature, **{label: float(coefficients[j, i]) for j, label in enumerate(labels)}}
                       for i, feature in enumerate(FEATURE_NAMES)])
    elif model_name == "majority":
        chosen = int(np.bincount(targets["train"]).argmax())
        probabilities = np.zeros((len(parts["test"]), len(labels)))
        probabilities[:, chosen] = 1.0
    elif model_name == "chance":
        # A concrete reproducible uniform-random realization; expected accuracy=1/K.
        random_predictions = np.random.default_rng(seed).integers(0, len(labels), size=len(parts["test"]))
        probabilities = np.zeros((len(parts["test"]), len(labels)))
        probabilities[np.arange(len(parts["test"])), random_predictions] = 1.0
    else:
        raise ValueError(f"Unknown baseline: {model_name}")
    metrics = classification_metrics(targets["test"], probabilities.argmax(axis=1), labels)
    metrics["bootstrap_accuracy_ci95"] = cluster_bootstrap_accuracy(targets["test"], probabilities.argmax(axis=1),
                                                                   [row.group_id for row in parts["test"]])
    result = {"run_id": run_id, "experiment": experiment, "model": model_name, "mode": mode,
              "seed": seed, "variant": "raw", "heldout_domain": heldout_domain,
              "heldout_source": heldout_source,
              "dataset_sha256": dataset_sha256, "split_sha256": split_fingerprint(split),
              "implementation_sha256": implementation_fingerprint(), "labels": labels, **metrics,
              "train_rows": len(parts["train"]), "validation_rows": len(parts["validation"]),
              "best_validation_macro_f1": best_val if best_val >= 0 else None,
              "selected_C": best_c, "expected_chance_accuracy": 1.0 / len(labels),
              "seconds": time.perf_counter() - started, "test_prompt_ids": split["test"]}
    result["split_group_counts"] = {name: len({row.group_id for row in part}) for name, part in parts.items()}
    result["excluded_rows"] = len(parts.get("excluded", []))
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "run_config.json", result)
    write_csv(run_dir / "predictions.csv", prediction_records(parts["test"], probabilities, labels))
    print(f"DONE {run_id} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
    return result

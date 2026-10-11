"""Run the assignment's required comparisons and optional domain/style studies."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_attribution.data import MODES, fingerprint_file, load_rows, split_prompts, validate_rows
from llm_attribution.experiment import (TrainingConfig, environment_info, run_baseline,
                                      run_neural, write_csv, write_json)


def aggregate_results(out: Path, runs: list[dict], metadata: dict) -> None:
    write_json(out / "results.json", {"metadata": metadata, "runs": runs})
    fields = ["run_id", "experiment", "model", "mode", "variant", "seed", "heldout_domain", "heldout_source",
              "accuracy", "macro_f1", "n_test_rows", "train_rows", "validation_rows",
              "best_epoch", "epochs_run", "best_validation_macro_f1", "selected_C", "seconds"]
    write_csv(out / "results.csv", [{key: run.get(key) for key in fields} for run in runs], fields)
    aggregate = []
    for model in ("cnn", "lstm"):
        for mode in MODES:
            matching = [run for run in runs if run["experiment"] == "main" and run["model"] == model and run["mode"] == mode]
            if not matching:
                continue
            import numpy as np
            aggregate.append({"model": model, "mode": mode, "n_seeds": len(matching),
                              "accuracy_mean": float(np.mean([run["accuracy"] for run in matching])),
                              "accuracy_std": float(np.std([run["accuracy"] for run in matching], ddof=1)) if len(matching) > 1 else 0.0,
                              "macro_f1_mean": float(np.mean([run["macro_f1"] for run in matching])),
                              "macro_f1_std": float(np.std([run["macro_f1"] for run in matching], ddof=1)) if len(matching) > 1 else 0.0})
    write_csv(out / "main_aggregate.csv", aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--heldout-domain", default=None)
    parser.add_argument("--heldout-source", default="selfinstruct")
    parser.add_argument("--skip-domain", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument("--models", nargs="+", choices=["cnn", "lstm"], default=["cnn", "lstm"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    rows = load_rows(args.data)
    dataset = validate_rows(rows)
    dataset_sha256 = fingerprint_file(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            max_length=args.max_length, threads=args.threads)
    split = split_prompts(rows, seed=args.split_seed)
    write_json(args.output / "split.json", {"dataset_sha256": dataset_sha256, "split_seed": args.split_seed,
                                           "split_unit": "split_group_id (near-duplicate components)", **split})
    metadata = {"dataset": dataset, "dataset_sha256": dataset_sha256, "split_seed": args.split_seed,
                "training_seeds": args.seeds, "training_config": asdict(config), "environment": environment_info(),
                "split_counts_prompts": {name: len(ids) for name, ids in split.items()},
                "split_counts_groups": {name: len({row.group_id for row in rows if row.prompt_id in set(ids)})
                                        for name, ids in split.items()},
                "notes": ["All answers for a prompt and its near-duplicate template component remain in one split.",
                          "Training seeds vary initialization/shuffling only; split membership stays fixed.",
                          "Vocabulary and TF-IDF are fit on training data only.",
                          "Model selection uses validation macro-F1 only; test metrics are computed after selection.",
                          "Bootstrap intervals resample entire template components and describe test-sample uncertainty.",
                          "Neural output budget256 is shared by output and combined conditions; combined prepends up to64 question tokens and a separator (max321).",
                          "TF-IDF baselines consume full texts without the neural token budgets.",
                          "Domain and ablation experiments use seed42 and are exploratory."]}
    write_json(args.output / "experiment_metadata.json", metadata)
    runs = []
    for mode in args.modes:
        runs.append(run_baseline(rows, split, args.output, "tfidf", mode, 42, dataset_sha256))
    for baseline in ("features", "chance", "majority"):
        runs.append(run_baseline(rows, split, args.output, baseline, "output", 42, dataset_sha256))
    aggregate_results(args.output, runs, metadata)
    if not args.only_baselines:
        for seed in args.seeds:
            for model in args.models:
                for mode in args.modes:
                    runs.append(run_neural(rows, split, args.output, model, mode, seed, config,
                                           dataset_sha256, force=args.force))
                    aggregate_results(args.output, runs, metadata)
    if not args.skip_domain:
        heldout_domain = args.heldout_domain
        heldout_source = None if heldout_domain else args.heldout_source
        target = heldout_domain or heldout_source
        if target:
            domain_split = split_prompts(rows, seed=args.split_seed, holdout_domain=heldout_domain,
                                         holdout_source=heldout_source)
            write_json(args.output / f"domain_split_{target}.json", {
                "dataset_sha256": dataset_sha256, "split_seed": args.split_seed,
                "heldout_domain": heldout_domain, "heldout_source": heldout_source,
                "holdout_definition": "verified upstream prompt source" if heldout_source else "heuristic semantic category",
                "purge_rule": "non-target-source prompts sharing a target test template are excluded", **domain_split})
            for mode in args.modes:
                runs.append(run_baseline(rows, domain_split, args.output, "tfidf", mode, 42,
                                         dataset_sha256, "domain_holdout", heldout_domain, heldout_source))
            if not args.only_baselines:
                for model in args.models:
                    runs.append(run_neural(rows, domain_split, args.output, model, "output", 42, config,
                                           dataset_sha256, "domain_holdout", heldout_domain=heldout_domain,
                                           force=args.force, heldout_source=heldout_source))
            metadata["heldout_domain"] = heldout_domain
            metadata["heldout_source"] = heldout_source
            metadata["domain_split_counts_prompts"] = {name: len(ids) for name, ids in domain_split.items()}
        else:
            metadata["domain_holdout_skipped_reason"] = "No named domain contains at least35 matched prompts"
        aggregate_results(args.output, runs, metadata)
    if not args.skip_ablation and not args.only_baselines:
        for variant in ("length64", "normalized"):
            for model in args.models:
                runs.append(run_neural(rows, split, args.output, model, "output", 42, config,
                                       dataset_sha256, "ablation", variant=variant, force=args.force))
                aggregate_results(args.output, runs, metadata)
    aggregate_results(args.output, runs, metadata)
    print(f"Saved {len(runs)} completed runs to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

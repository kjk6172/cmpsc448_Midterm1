"""Stage and merge a fixed secondary RQ3 study: writing-domain holdout.

Staging reads the primary run metadata and writes only to .tools/writing_holdout.
Merge the five completed staged runs after the primary experiment has finished.
No hyperparameter or checkpoint choice is based on held-out writing results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_attribution.data import MODES, fingerprint_file, load_rows, split_prompts
from llm_attribution.experiment import (
    TrainingConfig, implementation_fingerprint, run_baseline, run_neural,
    split_fingerprint, write_json,
)
from run_experiments import aggregate_results

DOMAIN = "writing"
SEED = 42
EXPECTED_STAGE_IDS = {
    *(f"domain_holdout_{DOMAIN}_tfidf_{mode}_raw_seed{SEED}" for mode in MODES),
    *(f"domain_holdout_{DOMAIN}_{model}_output_raw_seed{SEED}" for model in ("cnn", "lstm")),
}


def read_primary(artifacts: Path, dataset_sha256: str) -> dict:
    path = artifacts / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Primary results are not available yet: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["metadata"]["dataset_sha256"] != dataset_sha256:
        raise ValueError("The primary results and current dataset have different SHA-256 values")
    if any(run["dataset_sha256"] != dataset_sha256 for run in result["runs"]):
        raise ValueError("Primary results contain a run from a different dataset")
    return result


def require_primary_complete(primary: dict) -> None:
    """Refuse to merge into an incomplete version of the prescribed main run."""
    runs = {run["run_id"] for run in primary["runs"]}
    metadata = primary["metadata"]
    required = {
        *(f"main_{model}_{mode}_raw_seed{seed}"
          for seed in metadata["training_seeds"]
          for model in ("cnn", "lstm") for mode in MODES),
        *(f"main_tfidf_{mode}_raw_seed42" for mode in MODES),
        *(f"main_{model}_output_raw_seed42" for model in ("features", "chance", "majority")),
        *(f"ablation_{model}_output_{variant}_seed42"
          for model in ("cnn", "lstm") for variant in ("length64", "normalized")),
    }
    source = metadata.get("heldout_source")
    semantic_domain = metadata.get("heldout_domain")
    if not source and not semantic_domain:
        raise ValueError("Primary RQ3 source/domain study has not finished; merge only after the primary process exits")
    target = source or semantic_domain
    required.update(f"domain_holdout_{target}_tfidf_{mode}_raw_seed42" for mode in MODES)
    required.update(f"domain_holdout_{target}_{model}_output_raw_seed42" for model in ("cnn", "lstm"))
    missing = required - runs
    if missing:
        raise ValueError(f"Primary prescribed experiments are incomplete ({len(missing)} runs missing); merge after completion")


def holdout_description(rows, split, config: TrainingConfig) -> dict:
    return {
        "heldout_domain": DOMAIN,
        "heldout_source": None,
        "holdout_definition": "Fixed input-only keyword heuristic: writing; semantic tags are approximate, not expert annotations.",
        "split_unit": "split_group_id (near-duplicate components)",
        "purge_rule": "Non-target prompts sharing a target test template component are excluded from training and validation.",
        "training_seed": SEED,
        "training_config": asdict(config),
        "split_sha256": split_fingerprint(split),
        "split_counts_prompts": {name: len(ids) for name, ids in split.items()},
        "split_counts_groups": {
            name: len({row.group_id for row in rows if row.prompt_id in set(ids)})
            for name, ids in split.items()
        },
        "run_ids": sorted(EXPECTED_STAGE_IDS),
        "interpretation": "Secondary exploratory semantic-domain generalization study, using unchanged main hyperparameters and validation-only selection.",
    }


def stage(data: Path, artifacts: Path, stage_dir: Path, force: bool) -> None:
    dataset_sha256 = fingerprint_file(data)
    primary = read_primary(artifacts, dataset_sha256)
    metadata = primary["metadata"]
    config = TrainingConfig(**metadata["training_config"])
    rows = load_rows(data)
    split = split_prompts(rows, seed=metadata["split_seed"], holdout_domain=DOMAIN)
    description = holdout_description(rows, split, config)
    split_record = {
        "dataset_sha256": dataset_sha256,
        "split_seed": metadata["split_seed"],
        **description,
        **split,
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / f"domain_split_{DOMAIN}.json", split_record)
    runs = []
    for mode in MODES:
        runs.append(run_baseline(rows, split, stage_dir, "tfidf", mode, SEED,
                                 dataset_sha256, "domain_holdout", DOMAIN))
    for model in ("cnn", "lstm"):
        runs.append(run_neural(rows, split, stage_dir, model, "output", SEED,
                               config, dataset_sha256, "domain_holdout",
                               heldout_domain=DOMAIN, force=force))
    if {run["run_id"] for run in runs} != EXPECTED_STAGE_IDS:
        raise ValueError("Secondary writing experiment did not produce precisely the five expected runs")
    stage_result = {
        "metadata": {
            "dataset_sha256": dataset_sha256,
            "primary_split_seed": metadata["split_seed"],
            "primary_training_config": asdict(config),
            "implementation_sha256": implementation_fingerprint(),
            "additional_domain_holdout": description,
            "stage_only": True,
        },
        "split": split_record,
        "runs": runs,
    }
    write_json(stage_dir / "stage_results.json", stage_result)
    print(f"Staged five completed writing-holdout runs in {stage_dir}; primary artifacts were not modified.", flush=True)


def merge(data: Path, artifacts: Path, stage_dir: Path) -> None:
    dataset_sha256 = fingerprint_file(data)
    primary = read_primary(artifacts, dataset_sha256)
    require_primary_complete(primary)
    staged = json.loads((stage_dir / "stage_results.json").read_text(encoding="utf-8"))
    stage_metadata = staged["metadata"]
    primary_metadata = primary["metadata"]
    if stage_metadata["dataset_sha256"] != dataset_sha256:
        raise ValueError("Staged results belong to a different dataset")
    if stage_metadata["primary_training_config"] != primary_metadata["training_config"]:
        raise ValueError("Staged and primary experiments used different hyperparameters")
    if stage_metadata["primary_split_seed"] != primary_metadata["split_seed"]:
        raise ValueError("Staged and primary experiments used different split seeds")
    if stage_metadata["implementation_sha256"] != implementation_fingerprint():
        raise ValueError("The ML implementation changed after staging; restage before merging")
    if {run["run_id"] for run in staged["runs"]} != EXPECTED_STAGE_IDS or len(staged["runs"]) != 5:
        raise ValueError("Stage package is missing or duplicating writing experiment runs")
    expected_split = split_prompts(load_rows(data), seed=primary_metadata["split_seed"], holdout_domain=DOMAIN)
    expected_split_sha256 = split_fingerprint(expected_split)
    for run in staged["runs"]:
        if (run["dataset_sha256"] != dataset_sha256
                or run["split_sha256"] != expected_split_sha256
                or run["implementation_sha256"] != stage_metadata["implementation_sha256"]):
            raise ValueError(f"Staged run has inconsistent provenance: {run['run_id']}")
        source = stage_dir / "runs" / run["run_id"]
        required_files = ["metrics.json", "run_config.json", "predictions.csv"]
        if run["model"] in ("cnn", "lstm"):
            required_files += ["checkpoint.pt", "vocab.json", "history.csv"]
        if not all((source / filename).is_file() for filename in required_files):
            raise ValueError(f"Staged run artifacts are incomplete: {source}")
        saved = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        if saved != run:
            raise ValueError(f"Staged aggregate metrics disagree with the saved run: {run['run_id']}")
    # All provenance and completion checks precede mutations of primary artifacts.
    for run in staged["runs"]:
        shutil.copytree(stage_dir / "runs" / run["run_id"],
                        artifacts / "runs" / run["run_id"], dirs_exist_ok=True)
    write_json(artifacts / f"domain_split_{DOMAIN}.json", staged["split"])
    combined = {run["run_id"]: run for run in primary["runs"]}
    combined.update({run["run_id"]: run for run in staged["runs"]})
    primary_metadata.setdefault("additional_domain_holdouts", {})[DOMAIN] = stage_metadata["additional_domain_holdout"]
    aggregate_results(artifacts, list(combined.values()), primary_metadata)
    write_json(artifacts / "experiment_metadata.json", primary_metadata)
    print(f"Merged writing holdout; {len(combined)} unique completed runs are preserved in {artifacts / 'results.json'}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "dataset.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--stage-dir", type=Path, default=ROOT / ".tools" / "writing_holdout")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage-only", action="store_true", help="Run the secondary study without writing primary artifacts")
    action.add_argument("--merge-only", action="store_true", help="Merge an existing completed stage after primary completion")
    parser.add_argument("--force", action="store_true", help="Retrain already cached secondary neural runs")
    args = parser.parse_args()
    if args.merge_only:
        merge(args.data, args.artifacts, args.stage_dir)
    elif args.stage_only:
        stage(args.data, args.artifacts, args.stage_dir, args.force)
    else:
        primary = read_primary(args.artifacts, fingerprint_file(args.data))
        require_primary_complete(primary)
        stage(args.data, args.artifacts, args.stage_dir, args.force)
        merge(args.data, args.artifacts, args.stage_dir)


if __name__ == "__main__":
    main()

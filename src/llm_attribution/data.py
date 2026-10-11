"""Dataset checks, leakage-resistant splits, and train-only tokenization."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MODES = ("input", "output", "input_output")


@dataclass(frozen=True)
class Row:
    prompt_id: str
    llm_name: str
    llm_input: str
    llm_output: str
    domain: str
    source_dataset: str = "unspecified"
    split_group_id: str = ""

    @property
    def group_id(self) -> str:
        return self.split_group_id or self.prompt_id

    def text(self, mode: str, normalize: bool = False) -> str:
        output = normalize_formatting(self.llm_output) if normalize else self.llm_output
        if mode == "input":
            return self.llm_input
        if mode == "output":
            return output
        if mode == "input_output":
            return self.llm_input + " [SEP] " + output
        raise ValueError(f"Unsupported input mode: {mode}")


def normalize_formatting(text: str) -> str:
    """Remove common Markdown delimiters without changing lexical content."""
    text = re.sub(r"```[^\n]*\n?", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"[`*_~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    result = []
    for part in re.split(r"(\[SEP\])", text):
        result.extend(["[SEP]"] if part == "[SEP]" else TOKEN_RE.findall(part.lower()))
    return result


def load_rows(path: str | Path) -> list[Row]:
    rows: list[Row] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {str(field).lower(): field for field in reader.fieldnames or []}
        required = ("prompt_id", "llm_name", "llm_input", "llm_output")
        missing = set(required) - fields.keys()
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            values = {name: (raw[fields[name]] or "").strip() for name in required}
            domain = (raw.get(fields.get("domain", "domain")) or "unspecified").strip()
            source = (raw.get(fields.get("source_dataset", "source_dataset")) or "unspecified").strip()
            group_id = (raw.get(fields.get("split_group_id", "split_group_id")) or values["prompt_id"]).strip()
            rows.append(Row(**values, domain=domain, source_dataset=source, split_group_id=group_id))
    validate_rows(rows)
    return rows


def validate_rows(rows: list[Row]) -> dict:
    if not rows:
        raise ValueError("Dataset is empty")
    labels = sorted({row.llm_name for row in rows})
    if len(labels) < 3:
        raise ValueError("Assignment requires at least three model families")
    by_prompt: dict[str, list[Row]] = defaultdict(list)
    input_to_id: dict[str, str] = {}
    for row in rows:
        if not all((row.prompt_id, row.llm_name, row.llm_input, row.llm_output)):
            raise ValueError("Every row needs a prompt ID, label, question, and nonempty answer")
        by_prompt[row.prompt_id].append(row)
        input_fingerprint = re.sub(r"\s+", " ", row.llm_input.lower()).strip()
        previous = input_to_id.setdefault(input_fingerprint, row.prompt_id)
        if previous != row.prompt_id:
            raise ValueError(f"Identical question occurs under IDs {previous} and {row.prompt_id}; merge before splitting")
    for prompt_id, group in by_prompt.items():
        if len(group) != len(labels) or sorted(row.llm_name for row in group) != labels:
            raise ValueError(f"Prompt {prompt_id} does not contain exactly one answer from each family")
        if any(len({getattr(row, attribute) for row in group}) != 1
               for attribute in ("llm_input", "domain", "source_dataset", "group_id")):
            raise ValueError(f"Prompt {prompt_id} has inconsistent questions/domains")
    return {
        "rows": len(rows), "prompts": len(by_prompt), "labels": labels,
        "rows_per_label": dict(Counter(row.llm_name for row in rows)),
        "prompts_per_domain": dict(Counter(group[0].domain for group in by_prompt.values())),
        "prompts_per_source": dict(Counter(group[0].source_dataset for group in by_prompt.values())),
        "split_groups": len({row.group_id for row in rows}),
    }


def fingerprint_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_prompts(rows: list[Row], seed: int = 2026, test_fraction: float = 0.2,
                  val_fraction: float = 0.15, holdout_domain: str | None = None,
                  holdout_source: str | None = None) -> dict[str, list[str]]:
    """Keep near-duplicate prompt templates together, then expand prompt IDs."""
    prompts = {row.prompt_id: row for row in rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for pid, row in prompts.items():
        groups[row.group_id].append(pid)
    group_sources = {gid: Counter(prompts[pid].source_dataset for pid in pids).most_common(1)[0][0]
                     for gid, pids in groups.items()}
    group_ids = sorted(groups)
    test_ids, excluded = [], []
    if holdout_domain or holdout_source:
        attribute, value = ("source_dataset", holdout_source) if holdout_source else ("domain", holdout_domain)
        target_ids = {pid for pid, row in prompts.items() if getattr(row, attribute) == value}
        target_groups = {prompts[pid].group_id for pid in target_ids}
        test_ids = sorted(target_ids)
        excluded = sorted(pid for gid in target_groups for pid in groups[gid] if pid not in target_ids)
        remaining = [gid for gid in group_ids if gid not in target_groups]
        if not test_ids or len(remaining) < 4:
            raise ValueError(f"Insufficient data for holdout {attribute}={value}")
    else:
        strata = [group_sources[gid] for gid in group_ids]
        counts = Counter(strata)
        stratify = strata if min(counts.values()) >= 3 and int(len(group_ids) * test_fraction) >= len(counts) else None
        remaining, test_groups = train_test_split(group_ids, test_size=test_fraction, random_state=seed, stratify=stratify)
        test_ids = sorted(pid for gid in test_groups for pid in groups[gid])
    conditional_val = val_fraction / (1.0 - test_fraction)
    strata = [group_sources[gid] for gid in remaining]
    counts = Counter(strata)
    stratify = strata if min(counts.values()) >= 2 and int(len(remaining) * conditional_val) >= len(counts) else None
    train_groups, val_groups = train_test_split(remaining, test_size=conditional_val, random_state=seed + 1, stratify=stratify)
    split = {"train": sorted(pid for gid in train_groups for pid in groups[gid]),
             "validation": sorted(pid for gid in val_groups for pid in groups[gid]), "test": test_ids}
    if excluded:
        split["excluded"] = excluded
    verify_split(rows, split)
    return split


def verify_split(rows: list[Row], split: dict[str, list[str]]) -> None:
    parts = [set(split[key]) for key in split]
    if any(parts[i] & parts[j] for i in range(len(parts)) for j in range(i + 1, len(parts))):
        raise ValueError("Prompt leakage between data splits")
    if set.union(*parts) != {row.prompt_id for row in rows}:
        raise ValueError("Split does not cover precisely the dataset prompt IDs")
    if any(not split[key] for key in ("train", "validation", "test")):
        raise ValueError("Train, validation, and test splits must be nonempty")
    prompt_groups = {row.prompt_id: row.group_id for row in rows}
    group_parts = [{prompt_groups[pid] for pid in split[key]} for key in ("train", "validation", "test")]
    if any(group_parts[i] & group_parts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Near-duplicate template group leaks between splits")


def select_rows(rows: list[Row], ids: Iterable[str]) -> list[Row]:
    membership = set(ids)
    return sorted((row for row in rows if row.prompt_id in membership), key=lambda row: (row.prompt_id, row.llm_name))


class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.stoi = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def fit(cls, texts: Iterable[str], max_vocab: int = 20000, min_frequency: int = 2) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        ordered = sorted((token for token, count in counts.items() if count >= min_frequency and token != "[SEP]"),
                         key=lambda token: (-counts[token], token))
        return cls(["[PAD]", "[UNK]", "[SEP]"] + ordered[:max(0, max_vocab - 3)])

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.stoi.get(token, 1) for token in tokenize(text)[:max_length]]
        return ids or [1]

    def encode_row(self, row: Row, mode: str, output_budget: int, input_budget: int = 64,
                   normalize: bool = False) -> list[int]:
        if mode == "input":
            return self.encode(row.llm_input, input_budget)
        output = normalize_formatting(row.llm_output) if normalize else row.llm_output
        answer = self.encode(output, output_budget)
        if mode == "output":
            return answer
        if mode == "input_output":
            return self.encode(row.llm_input, input_budget) + [self.stoi["[SEP]"]] + answer
        raise ValueError(f"Unsupported input mode: {mode}")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")


class TextDataset(Dataset):
    def __init__(self, rows: list[Row], mode: str, vocab: Vocabulary, label_to_id: dict[str, int],
                 max_length: int, normalize: bool = False, input_budget: int = 64):
        self.rows = rows
        self.encoded = [(vocab.encode_row(row, mode, max_length, input_budget, normalize), label_to_id[row.llm_name]) for row in rows]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int):
        return self.encoded[index]


def collate_batch(batch):
    tokens, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in tokens], dtype=torch.long)
    padded = torch.zeros((len(tokens), max(5, int(lengths.max()))), dtype=torch.long)
    for index, ids in enumerate(tokens):
        padded[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def handcrafted_features(texts: Iterable[str]) -> np.ndarray:
    result = []
    for text in texts:
        tokens = tokenize(text)
        words = re.findall(r"\b\w+\b", text.lower())
        lines = text.splitlines()
        n_chars = max(1, len(text))
        result.append([
            np.log1p(len(tokens)), np.log1p(len(text)), np.log1p(len(lines)),
            len(set(words)) / max(1, len(words)),
            sum(len(word) for word in words) / max(1, len(words)),
            text.count("\n") / n_chars, text.count("*") / n_chars,
            text.count("#") / n_chars, text.count("`") / n_chars,
            text.count(":") / n_chars, text.count(";") / n_chars,
            text.count("!") / n_chars, text.count("?") / n_chars,
            sum(c.isdigit() for c in text) / n_chars,
            sum(bool(re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s)", line)) for line in lines) / max(1, len(lines)),
            len(re.findall(r"```", text)),
        ])
    return np.asarray(result, dtype=np.float64)


FEATURE_NAMES = ["log_tokens", "log_characters", "log_lines", "type_token_ratio", "mean_word_length",
                 "newline_rate", "asterisk_rate", "hash_rate", "backtick_rate", "colon_rate", "semicolon_rate",
                 "exclamation_rate", "question_rate", "digit_rate", "list_line_rate", "code_fence_count"]

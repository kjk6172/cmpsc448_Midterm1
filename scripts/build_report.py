"""Build a submission report and figures exclusively from saved experiment outputs.

Run from the repository root: python scripts/build_report.py
No numerical result is entered by hand. Missing measurements are a build error.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_attribution.data import tokenize, normalize_formatting  # noqa: E402

NAVY = "#18324F"
TEAL = "#167D8D"
ORANGE = "#D77736"
GRAY = "#536275"
MODEL_NAMES = {"cnn": "CNN", "lstm": "BiLSTM", "tfidf": "TF-IDF", "features": "Style features", "chance": "Chance", "majority": "Majority"}
MODE_NAMES = {"input": "Prompt only", "output": "Response only", "input_output": "Prompt + response"}
VARIANT_NAMES = {"raw": "Original", "length64": "First 64 tokens", "normalized": "Formatting normalized"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, percent=True, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    value = float(value) * (100 if percent else 1)
    return f"{value:.{digits}f}" + ("%" if percent else "")


def interval(values, percent=True):
    return "[" + ", ".join(fmt(x, percent) for x in values) + "]"


def field(run, key, default=None):
    return run.get(key, default)


def holdout_name(run):
    return str(run.get("heldout_source") or run.get("heldout_domain") or "unspecified")


def comparison_label(name):
    replacements = {
        "cnn_input_output_minus_output": "CNN: combined - response only",
        "lstm_input_output_minus_output": "BiLSTM: combined - response only",
        "cnn_minus_lstm_output": "Response only: CNN - BiLSTM",
        "cnn_normalized_minus_raw": "CNN: normalized - original (seed 42)",
        "lstm_normalized_minus_raw": "BiLSTM: normalized - original (seed 42)",
        "cnn_length64_minus_raw": "CNN: 64 tokens - original (seed 42)",
        "lstm_length64_minus_raw": "BiLSTM: 64 tokens - original (seed 42)",
    }
    return replacements.get(name,name)


def raw_main(run):
    return run.get("experiment") == "main" and run.get("variant", "raw") == "raw"


def aggregate(runs):
    grouped = {}
    for run in runs:
        key = (run["model"], run["mode"])
        grouped.setdefault(key, []).append(run)
    result = []
    for (model, mode), group in grouped.items():
        item = {"model": model, "mode": mode, "n": len(group), "runs": group}
        for metric in ("accuracy", "macro_f1"):
            vals = [float(run[metric]) for run in group]
            item[metric] = statistics.mean(vals)
            item[metric + "_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        result.append(item)
    return sorted(result, key=lambda x: (list(MODE_NAMES).index(x["mode"]), list(MODEL_NAMES).index(x["model"])))


def metric_mean_std(item, name):
    std = item[name + "_std"]
    return fmt(item[name]) + (f" +/- {float(std)*100:.1f} pp" if std is not None else " (one run)")


def run_predictions(run):
    return pd.read_csv(ROOT / "artifacts" / "runs" / run["run_id"] / "predictions.csv")


def find_dataset():
    candidates = [ROOT / "data" / "processed" / "dataset.csv", ROOT / "data" / "processed" / "llm_responses.csv", ROOT / "data" / "curated.csv", ROOT / "data" / "dataset.csv"]
    candidates.extend(sorted((ROOT / "data").glob("**/*.csv")))
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, encoding="utf-8-sig")
            names = {str(c).lower(): c for c in df.columns}
            if {"prompt_id", "llm_name", "llm_input", "llm_output"}.issubset(names):
                return candidate, df.rename(columns={v: k for k, v in names.items()})
    raise FileNotFoundError("No curated data CSV with prompt_id / LLM_name / LLM_Input / LLM_output")


def find_manifest():
    for name in ("data/source_manifest.json", "data/manifest.json", "data/processed/manifest.json", "data/provenance.json"):
        path = ROOT / name
        if path.exists():
            return read_json(path)
    raise FileNotFoundError("A dataset provenance manifest is required")


def setup_plotting():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelcolor": NAVY, "text.color": NAVY, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .17, "savefig.dpi": 190, "figure.facecolor": "white"})


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(runs, data, figure_dir):
    setup_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)
    figs = {}
    main = [r for r in runs if raw_main(r)]
    summary = aggregate([r for r in main if r["model"] in ("cnn", "lstm")])
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    x = np.arange(3)
    for j, model in enumerate(("cnn", "lstm")):
        items = [next(i for i in summary if i["model"] == model and i["mode"] == mode) for mode in MODE_NAMES]
        y = [100*i["accuracy"] for i in items]
        err = [100*(i["accuracy_std"] or 0) for i in items]
        bars = ax.bar(x + (j-.5)*.32, y, .30, yerr=err, capsize=4, label=MODEL_NAMES[model], color=(TEAL, ORANGE)[j])
        for bar, value in zip(bars, y):
            ax.text(bar.get_x()+bar.get_width()/2, min(100, value+3), f"{value:.1f}", ha="center", fontsize=9)
    ax.axhline(100/3, ls="--", color=GRAY, lw=1, label="Chance (33.3%)")
    ax.set_xticks(x, list(MODE_NAMES.values()))
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.set_title("Same questions, same split: the input comparison")
    figs["main"] = figure_dir / "main_comparison.png"
    save_fig(fig, figs["main"])

    reps = [next(r for r in main if r["model"] == model and r["mode"] == "output" and r["seed"] == 42) for model in ("cnn", "lstm")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for ax, run in zip(axes, reps):
        matrix = np.array(run["confusion_matrix"])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(run["labels"])):
            for j in range(len(run["labels"])):
                ax.text(j, i, f"{matrix[i,j]}\n{normalized[i,j]*100:.0f}%", ha="center", va="center", fontsize=9, color="white" if normalized[i,j] > .55 else NAVY)
        labels = [str(label).split("-")[0] if len(str(label)) > 12 else str(label) for label in run["labels"]]
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(False)
        ax.set_xlabel("Predicted family")
        ax.set_ylabel("True family")
        ax.set_title(MODEL_NAMES[run["model"]]+" | seed 42")
    fig.tight_layout()
    figs["confusion"] = figure_dir / "output_confusions.png"
    save_fig(fig, figs["confusion"])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for ax, run in zip(axes, reps):
        history = pd.read_csv(ROOT / "artifacts" / "runs" / run["run_id"] / "history.csv")
        ax.plot(history["epoch"], history["train_loss"], "o-", ms=3, label="Train loss", color=TEAL)
        ax.plot(history["epoch"], history["validation_loss"], "o-", ms=3, label="Validation loss", color=ORANGE)
        ax.axvline(run["best_epoch"], ls="--", color=GRAY, lw=1, label="Selected epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy")
        ax.set_title(MODEL_NAMES[run["model"]]+" | response only")
        ax.legend(fontsize=8)
    fig.tight_layout()
    figs["learning"] = figure_dir / "learning_curves.png"
    save_fig(fig, figs["learning"])

    lengths = data.assign(n_words=data.llm_output.map(lambda t: len(re.findall(r"\b\w+\b", str(t)))))
    labels = sorted(data.llm_name.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
    axes[0].boxplot([lengths.loc[lengths.llm_name == label, "n_words"].to_numpy() for label in labels], tick_labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "#C8DFE5"}, medianprops={"color": ORANGE})
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Response words (log scale)")
    axes[0].set_title("Test response lengths")
    def prefix_ttr(text):
        words = re.findall(r"\b\w+\b", str(text).lower())[:64]
        return len(set(words))/max(len(words), 1)
    lengths["prefix_ttr"] = lengths.llm_output.map(prefix_ttr)
    vals = [lengths.loc[lengths.llm_name == label, "prefix_ttr"].mean() for label in labels]
    axes[1].bar(labels, vals, color=[TEAL, ORANGE, "#597A9D"][:len(labels)])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean unique / total words")
    axes[1].set_title("First 64-word diversity")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    figs["features"] = figure_dir / "response_features.png"
    save_fig(fig, figs["features"])

    domain_runs = [r for r in runs if r["experiment"] == "domain_holdout" and r["model"] in ("cnn", "lstm") and r["mode"] == "output"]
    if domain_runs:
        fig, ax = plt.subplots(figsize=(7.1, 2.7))
        positions, vals, colors_ = [], [], []
        names = []
        for j, model in enumerate(("cnn", "lstm")):
            overall = next(s for s in summary if s["model"] == model and s["mode"] == "output")
            names.append(MODEL_NAMES[model]+"\nMixed-source test")
            vals.append(overall["accuracy"]*100)
            colors_.append((TEAL, ORANGE)[j])
            for run in [r for r in domain_runs if r["model"] == model]:
                names.append(MODEL_NAMES[model]+"\nHoldout: "+holdout_name(run))
                vals.append(run["accuracy"]*100)
                colors_.append((TEAL, ORANGE)[j])
        bars = ax.bar(range(len(names)), vals, color=colors_, alpha=.9)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.axhline(100/3, color=GRAY, ls="--", lw=1)
        for b, val in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, val+2, f"{val:.1f}", ha="center", fontsize=9)
        ax.set_title("RQ3: generalization to a held-out prompt source")
        fig.tight_layout()
        figs["domain"] = figure_dir / "domain_generalization.png"
        save_fig(fig, figs["domain"])
    writing = [r for r in runs if r["experiment"] == "domain_holdout" and r.get("heldout_domain") == "writing" and r["model"] in ("cnn","lstm")]
    if len(writing) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
        for ax, run in zip(axes,sorted(writing,key=lambda r:r["model"])):
            matrix = np.array(run["confusion_matrix"])
            normalized = matrix / np.maximum(matrix.sum(axis=1,keepdims=True),1)
            ax.imshow(normalized,vmin=0,vmax=1,cmap="Blues")
            for i in range(len(run["labels"])):
                for j in range(len(run["labels"])):
                    ax.text(j,i,f"{matrix[i,j]}\n{normalized[i,j]*100:.0f}%",ha="center",va="center",fontsize=9,color="white" if normalized[i,j]>.55 else NAVY)
            ax.set_xticks(range(3),run["labels"],rotation=15)
            ax.set_yticks(range(3),run["labels"])
            ax.set_xlabel("Predicted family")
            ax.set_ylabel("True family")
            ax.grid(False)
            ax.set_title(MODEL_NAMES[run["model"]]+" | writing holdout")
        fig.tight_layout()
        figs["writing_confusion"] = figure_dir / "writing_holdout_confusions.png"
        save_fig(fig,figs["writing_confusion"])
    return figs


class Report:
    def __init__(self, metadata, outdir):
        self.metadata = metadata
        self.outdir = outdir
        self.story = []
        self.md = []
        base = getSampleStyleSheet()
        self.styles = {
            "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13.2, spaceAfter=7, textColor=colors.HexColor(NAVY)),
            "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=colors.HexColor(GRAY), spaceAfter=6),
            "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor(NAVY), spaceAfter=14),
            "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor(TEAL), spaceBefore=8, spaceAfter=6),
            "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=30, leading=36, textColor=colors.HexColor(NAVY), spaceAfter=15),
            "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor(TEAL), spaceAfter=14),
            "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.8, leading=10.2, textColor=colors.HexColor(NAVY)),
            "headcell": ParagraphStyle("headcell", fontName="Helvetica-Bold", fontSize=7.8, leading=10.2, textColor=colors.white),
        }

    def paragraph(self, text, style="body"):
        self.story.append(Paragraph(html.escape(str(text)), self.styles[style]))
        self.md.append(str(text)+"\n")

    def heading(self, text, level=1):
        self.story.append(Paragraph(html.escape(text), self.styles["h1" if level == 1 else "h2"]))
        self.md.append("#"*(level+1)+" "+text+"\n")

    def page(self, title):
        if self.story:
            self.story.append(PageBreak())
        self.heading(title)

    def table(self, header, rows, widths=None):
        contents = [[Paragraph(html.escape(str(t)), self.styles["headcell"]) for t in header]]
        contents.extend([[Paragraph(html.escape(str(t)), self.styles["cell"]) for t in row] for row in rows])
        widths = widths or [516/len(header)]*len(header)
        table = Table(contents, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(NAVY)), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F0F5F8"), colors.white]), ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor(TEAL)), ("LINEBELOW", (0,-1), (-1,-1), .4, colors.HexColor("#CFD8E2"))]))
        self.story.extend([table, Spacer(1, 9)])
        self.md.append("| "+" | ".join(map(str,header))+" |\n| "+" | ".join("---" for _ in header)+" |")
        self.md.extend("| "+" | ".join(str(x).replace("|", "/") for x in row)+" |" for row in rows)
        self.md.append("")

    def figure(self, path, caption, height=None):
        pixel_width,pixel_height = ImageReader(str(path)).getSize()
        width = 516
        image = Image(str(path),width=width,height=height or width*pixel_height/pixel_width)
        self.story.append(KeepTogether([image, Paragraph(html.escape(caption), self.styles["small"]), Spacer(1, 7)]))
        self.md.append(f"![{caption}](figures/{path.name})\n")

    def build(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / "project_report.pdf"
        doc = SimpleDocTemplate(str(path), pagesize=(612,792), leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=44, title=self.metadata["title"], author=", ".join(self.metadata["authors"]), subject="Empirical LLM attribution with CNN and BiLSTM")
        def decorate(canvas, doc):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#D5DFE8"))
            canvas.line(48, 760, 564, 760)
            canvas.setFont("Helvetica", 7.4)
            canvas.setFillColor(colors.HexColor(GRAY))
            canvas.drawString(48, 770, "CMPSC 448  /  WHO WROTE IT?")
            canvas.drawRightString(564, 25, str(doc.page))
            canvas.drawString(48, 25, "  |  ".join(self.metadata["authors"])+"  |  "+self.metadata["submission_date"])
            canvas.restoreState()
        doc.build(self.story, onFirstPage=decorate, onLaterPages=decorate)
        (self.outdir / "project_report.md").write_text("\n".join(self.md), encoding="utf-8")
        return path


def data_feature_rows(data):
    rows = []
    for label, group in data.groupby("llm_name", sort=True):
        texts = group.llm_output.astype(str).tolist()
        words = [re.findall(r"\b\w+\b", t.lower()) for t in texts]
        counts = [len(w) for w in words]
        ttrs = [len(set(w))/max(1,len(w)) for w in words]
        prefix = [len(set(w[:64]))/max(1,len(w[:64])) for w in words]
        bullet = [bool(re.search(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s)",t)) for t in texts]
        fenced = ["```" in t for t in texts]
        rows.append([label, f"{np.mean(counts):.1f}", f"{np.median(counts):.0f}", f"{np.mean(ttrs):.3f}", f"{np.mean(prefix):.3f}", fmt(np.mean(bullet)), fmt(np.mean(fenced))])
    return rows


def normalize_results(raw):
    runs = raw if isinstance(raw, list) else raw.get("runs", raw.get("results", []))
    if not runs:
        raise ValueError("No completed experiment runs")
    needed = {(m, mode, seed) for m in ("cnn", "lstm") for mode in MODE_NAMES for seed in (42,43,44)}
    present = {(r["model"],r["mode"],r["seed"]) for r in runs if raw_main(r)}
    if not needed.issubset(present):
        raise ValueError(f"Missing required completed runs: {sorted(needed-present)}")
    for run in runs:
        if not (0 <= float(run["accuracy"]) <= 1 and 0 <= float(run["macro_f1"]) <= 1):
            raise ValueError(f"Invalid measured result in {run['run_id']}")
    return runs


def get_report_context():
    metadata = read_json(ROOT / "metadata.json")
    if not metadata.get("team_leader") or not metadata.get("team_members"):
        raise ValueError("Confirmed team leader and member metadata are required")
    results = read_json(ROOT / "artifacts" / "results.json")
    runs = normalize_results(results)
    dataset_path, data = find_dataset()
    manifest = find_manifest()
    main = [r for r in runs if raw_main(r)]
    representative = next(r for r in main if r["model"] == "cnn" and r["mode"] == "output" and r["seed"] == 42)
    test_ids = set(run_predictions(representative).prompt_id.astype(str))
    test_data = data[data.prompt_id.astype(str).isin(test_ids)].copy()
    if len(test_data) != int(representative["n_test_rows"]):
        raise ValueError("Report test features do not match the evaluated test rows")
    return metadata, results, runs, dataset_path, data, test_data, manifest


def build_report():
    metadata, result_document, runs, dataset_path, data, test_data, manifest = get_report_context()
    figures = make_figures(runs, test_data, ROOT / "report" / "figures")
    main = [r for r in runs if raw_main(r)]
    summary = aggregate(main)
    neural_summary = [s for s in summary if s["model"] in ("cnn", "lstm")]
    reps = {model: next(r for r in main if r["model"] == model and r["mode"] == "output" and r["seed"] == 42) for model in ("cnn", "lstm")}
    labels = sorted(data.llm_name.unique())
    counts = data.llm_name.value_counts().to_dict()
    prompt_count = data.prompt_id.nunique()
    domain_column = "source_dataset" if "source_dataset" in data else "domain"
    domain_counts = data.drop_duplicates("prompt_id")[domain_column].value_counts().to_dict()
    config = reps["cnn"].get("config", {})
    stat_path = ROOT / "artifacts" / "statistical_analysis.json"
    if not stat_path.exists():
        raise FileNotFoundError("Run scripts/analyze_results.py before building the final report")
    stat = read_json(stat_path)
    curation_audit = read_json(ROOT / "data" / "processed" / "audit.json")
    def comparison_summary(name, title):
        metric = next(c for c in stat["comparisons"] if c["comparison"] == name)["metrics"]["accuracy"]
        low, high = metric["ci95"]
        uncertainty = ("the interval includes zero, so a directional difference is not established"
                       if low <= 0 <= high else
                       "the interval excludes zero in this fixed test sample")
        return (f"{title}: {metric['delta']*100:+.1f} pp, paired 95% interval "
                f"[{low*100:+.1f}, {high*100:+.1f}] pp; {uncertainty}.")
    report = Report(metadata, ROOT / "report")

    # Page 1: main findings, not a decorative cover with no useful content.
    report.paragraph("WHO WROTE IT?", "title")
    report.paragraph("Identifying LLMs from Their Responses", "subtitle")
    report.paragraph("A matched-prompt study of CNN and recurrent neural text classifiers", "subtitle")
    report.table(["Team leader", "Team member", "Course / date"], [[metadata["team_leader"], ", ".join(metadata["team_members"]), metadata["course"]+" / "+metadata["submission_date"]]])
    report.heading("Abstract", 2)
    out = {m: next(s for s in neural_summary if s["model"] == m and s["mode"] == "output") for m in ("cnn","lstm")}
    combined = {m: next(s for s in neural_summary if s["model"] == m and s["mode"] == "input_output") for m in ("cnn","lstm")}
    report.paragraph(f"This project classifies the generating LLM family from public model responses. The curated corpus contains {len(data):,} responses to {prompt_count:,} distinct questions, with exactly one answer from each of {', '.join(labels)} for every question. The {data.split_group_id.nunique():,} exact/near-duplicate template components remain intact across partitions. We implement a text CNN and a bidirectional LSTM, compare prompt-only, response-only and combined inputs, and repeat the required neural experiments with three initialization seeds on one fixed grouped split. Response-only CNN and BiLSTM reach {fmt(out['cnn']['accuracy'])} and {fmt(out['lstm']['accuracy'])} mean test accuracy, respectively, against a 33.3% balanced-class chance reference.")
    report.paragraph(f"Adding the question changes mean accuracy by {(combined['cnn']['accuracy']-out['cnn']['accuracy'])*100:+.1f} percentage points for CNN and {(combined['lstm']['accuracy']-out['lstm']['accuracy'])*100:+.1f} points for BiLSTM. Prompt-only prediction acts as a controlled negative check: balanced identical questions cannot identify their three generating families. Additional experiments examine a held-out source, compact style features, formatting normalization and a 64-token input budget. The results support attribution only within the evaluated historical model versions and public benchmark distribution; they do not establish universal authorship detection.")
    report.heading("What is included", 2)
    report.table(["Requirement", "Evidence"], [["CNN and RNN implementations", "PyTorch CNN and packed-sequence BiLSTM; recorded configurations and checkpoints"], ["RQ1 and RQ2", "18 required neural runs: 2 models x 3 input conditions x 3 seeds"], ["RQ3 and RQ4", "Source-holdout evaluation, descriptive features, feature classifier and ablations"], ["Reproducibility", "Data manifest, fixed question IDs, metrics, predictions, histories and environment lock"]], [152,364])
    report.paragraph("All numerical results in this report are rendered from completed experiment files. No synthetic measurements or claimed student contribution split is inserted. AI assistance is disclosed with the interpretation and lessons.", "small")

    # Page 2: task and design.
    report.page("1. Problem and research design")
    report.paragraph("The supervised task is closed-set authorship attribution: given a question q, a response r, or both, predict one of three known LLM families. The class label is the generating family, not answer quality, factual correctness or whether the text was written by a human. Labels, generator identifiers and file metadata are excluded from the input text.")
    report.table(["Research question", "Operational comparison", "Evidence required"], [["RQ1: Who generated the response?", "CNN and BiLSTM on response only", "Accuracy, Macro-F1, class-level errors"], ["RQ2: Does the question help?", "Prompt / response / prompt + response, same split and seeds", "Within-model differences and prompt-only control"], ["RQ3: Does attribution transfer?", "Exclude one question-source domain from training", "Test on that source; qualify domain meaning"], ["RQ4: What signals distinguish models?", "Length/lexical/formatting descriptors; normalization; 64-token condition", "Interpret associations and ablation sensitivity cautiously"]], [133,209,174])
    report.heading("Controls that make the comparison meaningful", 2)
    report.paragraph("Each question is paired with all three families, keeping the question distribution and class counts exactly equal. All responses to one question, and all questions in its near-duplicate template component, remain in the same split. Every neural main condition uses the same partition and initialization seed list, while its vocabulary is fit only on its corresponding training text. The response-only and combined conditions both observe up to 256 response tokens; combined input prepends up to 64 question tokens. The output-only condition answers RQ1; the three conditions jointly answer RQ2.")
    report.heading("Hypotheses and decision rules", 2)
    report.paragraph("If response-only classification exceeds 1/3 accuracy, response-associated information helps on this benchmark. A combined-input gain suggests a possible benefit of context, but requires uncertainty analysis and cannot isolate context from changed sequence length, vocabulary and optimization. The prompt-only condition should be exactly 1/3 accurate, because a deterministic prediction for a repeated question can match exactly one of its three balanced labels. A source-holdout gap is evidence of distribution sensitivity, although a smaller training set and changed questions can also contribute.")
    report.paragraph("A positive classification result does not identify a causal linguistic fingerprint by itself. Generation settings, answer length, refusal style and benchmark construction may also carry family-associated signals. Normalization and fixed token budgets probe selected alternatives without proving that a neural network uses a unique mechanism.")

    # Page 3: provenance.
    report.page("2. Dataset provenance and curation")
    report.paragraph("We reuse existing model-output files from the public AlpacaEval repository rather than issue fresh API requests or fabricate responses. AlpacaEval was built as a model evaluation benchmark; this project repurposes its questions and generated outputs for attribution, without treating preference scores as authorship labels. Source files, upstream configurations, license notices, checksums and the transformation script are preserved in the repository.")
    report.table(["Family label", "Upstream generator", "Documented output budget", "Curated responses"], [["GPT", "gpt4_0613 / gpt-4-0613", "max_tokens = 4096", str(counts.get("GPT",counts.get("gpt",0)))], ["Claude", "claude-2", "max_tokens_to_sample = 4096", str(counts.get("Claude",counts.get("claude",0)))], ["Gemini", "gemini-pro", "max_output_tokens = 2048", str(counts.get("Gemini",counts.get("gemini",0)))]], [73,183,176,84])
    report.heading("Matched-question construction", 2)
    dropped = curation_audit["dropped_prompt_counts"]
    privacy_counts = {k.removeprefix("privacy_pattern_"): v for k,v in dropped.items() if k.startswith("privacy_pattern_")}
    report.paragraph(f"Responses are joined by a normalized question key. Starting from {curation_audit['original_matched_prompts']} matched questions, fixed rules exclude {dropped['excluded_crowdsourced_source']} OpenAssistant-source groups and all {dropped['excluded_koala_training_request']} Koala-source groups. Koala is excluded from training, validation and testing to honor its source README request against training use. After source exclusions, {dropped.get('empty_output',0)} blank-answer group and {sum(privacy_counts.values())} contact/address-pattern groups are removed ({', '.join(f'{v} {k}' for k,v in privacy_counts.items())}). These patterns can match fictional text and are screening criteria, not proof of real personal information. No answer quality or classifier score enters selection. The final {prompt_count} questions retain three nonempty responses each.")
    scrubbed = curation_audit["identity_scrubbed_output_counts"]
    report.paragraph(f"Case-insensitive model/provider substrings are replaced with the shared literal [MODEL] in {curation_audit['identity_scrubbed_input_prompts']} question groups and {sum(scrubbed.values())} answers ({', '.join(f'{v} {k}' for k,v in scrubbed.items())}). The script confirms identical cleaned inputs across families and no canonical duplicate questions. Stable question hashes and original-text hashes preserve traceability. Near-duplicate templates are grouped separately for evaluation; the answers themselves remain untruncated in the curated dataset.")
    report.table(["Retained prompt source / domain", "Distinct prompts", "Responses (3 per prompt)"], [[str(domain), str(count), str(count*3)] for domain,count in sorted(domain_counts.items())] + [["TOTAL", str(prompt_count), str(len(data))]], [288,110,118])
    report.heading("Rights and provenance", 2)
    report.paragraph("The upstream README explicitly distinguishes Apache-2.0 code and CC BY-NC 4.0 data. The processed dataset remains under the data license, with source attribution and retained notices for noncommercial academic use. The project's code license does not override those data terms. OpenAssistant-source exclusion is a conservative selection decision, not a claim that every possible source restriction has disappeared.")
    report.paragraph("Different upstream prompt wrappers and output budgets are important confounds: Gemini's documented cap is smaller than GPT/Claude's, and the historical model versions were not run under one identical decoding setup. Learned distinctions may therefore reflect collection conditions as well as model behavior. Data retrieval time is distinct from generation time.")
    report.paragraph("Primary source: https://github.com/tatsu-lab/alpaca_eval ; pinned revision cd543a149df89434d8a54582c0151c0b945c3d20; detailed immutable file URLs and SHA-256 hashes: data/source_manifest.json.", "small")

    # Page 4: leakage.
    report.page("3. Splits, leakage prevention and controls")
    reference = reps["cnn"]
    split_rows = [("Training", "train", reference["train_rows"]), ("Validation", "validation", reference["validation_rows"]), ("Test", "test", reference["n_test_rows"])]
    report.table(["Main partition", "Template clusters", "Distinct questions", "Response rows", "Each family"], [[name, str(reference["split_group_counts"][key]), str(int(n)//3), str(n), str(int(n)//3)] for name,key,n in split_rows], [130,106,106,90,84])
    report.paragraph("The main split targets 65% training, 15% validation and 20% testing at near-duplicate template-component level, with integer rounding. Prompt fractions differ slightly because cluster sizes vary. It is generated once with split seed 2026, using original source stratification where feasible. Model seeds 42, 43 and 44 change initialization and minibatch order, not test membership. Verification rejects overlap of both prompt IDs and template components, missing groups, duplicate normalized questions under different IDs and unbalanced groups.")
    grouping = curation_audit["grouping"]
    report.paragraph(f"The fixed input-only near-duplicate rule requires at least eight word tokens, a length ratio of at least 0.60, and either character SequenceMatcher similarity of at least 0.85 or word-trigram Jaccard similarity of at least 0.80. Connected components produce {grouping['split_group_count']} split groups from {prompt_count} questions; {grouping['multi_prompt_groups']} groups contain multiple questions and the largest contains {grouping['largest_group_prompts']} similar templates. This is stronger than exact-question grouping but remains a heuristic rather than complete semantic deduplication.")
    report.heading("What is fit on which data", 2)
    report.table(["Object", "Permitted data"], [["Token vocabulary and minimum frequency", "Training text for that input condition only"], ["TF-IDF vocabulary / IDF", "Training text only"], ["Feature scaling and classifier", "Training rows only"], ["Checkpoint selection / early stopping", "Validation Macro-F1 only"], ["Final accuracy, F1, errors and confidence intervals", "Test rows after model selection"], ["RQ4 descriptive plots", "Main test responses; descriptive analysis only"]], [251,265])
    report.heading("Why prompt-only accuracy is exactly chance", 2)
    report.paragraph("For every held-out question, the input-only model receives the same text three times, once for each family. In evaluation mode, each repeated text produces the same prediction. Exactly one row can be correct, yielding accuracy = number of questions / (3 x number of questions) = 1/3. This is a structural negative control, not a failed effort to optimize the classifier. Macro-F1 need not equal 1/3 because prediction frequencies can be uneven.")
    report.heading("Residual risks", 2)
    report.paragraph("Template-group splitting prevents exact and rule-detected near-duplicate question leakage, but does not guarantee that semantically similar questions cannot appear in separate partitions. The models may exploit content correctness, refusals, collection wrappers or formatting. Scrubbing explicit model/provider names removes direct labels but cannot remove every indirect identity phrase. The source-holdout study challenges one distribution shift; it cannot validate all future tasks or current model versions.")
    report.paragraph("The test set is shared across seed repetitions. Seed means describe training variability on one held-out sample; they are not results from three independently sampled test sets.", "small")

    # Page 5: actual implementation.
    report.page("4. CNN and recurrent model implementations")
    report.table(["Component", "CNN", "BiLSTM (required RNN)"], [["Embedding", "Trainable 64-dimensional embedding; PAD row", "Trainable 64-dimensional embedding; PAD row"], ["Encoder", "Parallel Conv1d kernels 3, 4, 5; 48 filters each; ReLU", "One bidirectional LSTM; 48 hidden units per direction"], ["Representation", "Masked global max pooling; concatenate 3 x 48 = 144 values", "Concatenate final forward and backward hidden states: 96 values"], ["Padding handling", "Fully contained windows; if shorter than kernel, one zero-completed window", "pack_padded_sequence ignores padded suffixes"], ["Classifier", "Dropout 0.3; linear 144 -> 3 logits", "Dropout 0.3; linear 96 -> 3 logits"], ["Loss", "Multiclass cross-entropy on logits", "Multiclass cross-entropy on logits"]], [110,203,203])
    report.heading("Tokenization and vocabulary", 2)
    report.paragraph("A reproducible regular expression extracts word-like units and individual non-whitespace punctuation after lowercasing. The prompt-response separator is reserved as [SEP]. The vocabulary includes PAD, UNK and SEP, retains training tokens appearing at least twice, and is capped at 20,000 entries. Unseen test tokens map to UNK. Embeddings are learned from scratch, so no pretrained language model or external embedding corpus contributes to these classifiers.")
    report.heading("Input construction and truncation", 2)
    report.paragraph("Prompt-only keeps at most 64 question tokens; response-only keeps at most 256 response tokens. Combined input concatenates the same capped 64-token question, one [SEP], and the same capped 256-token response, for at most 321 tokens. This preserves an equal response evidence budget when answering RQ2. Each batch is dynamically padded, with enough width for the largest CNN kernel. The combined model still has a longer sequence and a separately fitted vocabulary, so it is a pipeline comparison rather than a causal estimate of context alone.")
    audit = []
    for col,label,budget in (("llm_input","Question component",64),("llm_output","Response component",256)):
        lens = test_data[col].astype(str).map(lambda t: len(tokenize(t)))
        audit.append([label, str(budget), f"{lens.mean():.1f}", fmt((lens > budget).mean())])
    report.table(["Main test text", "Budget", "Mean original tokens", "Rows truncated"], audit, [184,63,143,126])
    report.paragraph("The CNN design is inspired by convolutional sentence classification (Kim, 2014); the recurrent model uses LSTM gating (Hochreiter and Schmidhuber, 1997). This implementation and its training setup are project code, not a claimed reproduction of either paper's benchmark results.", "small")

    # Page 6: training & metrics.
    report.page("5. Training, evaluation and uncertainty")
    report.table(["Setting", "Shared main neural protocol"], [["Optimizer", "AdamW; learning rate 0.002; weight decay 0.0001"], ["Batch size / gradient clipping", "32; global gradient norm clipped to 1.0"], ["Maximum epochs / early stopping", "20 epochs; patience 4 on validation Macro-F1"], ["Checkpoint selection", "Best validation Macro-F1, then lower validation loss in a tie; no test selection"], ["Repetitions", "Seeds 42, 43, 44; one fixed split seed 2026"], ["Vocabulary / standard response budget", "At most 20,000 vocabulary tokens; minimum count 2; first 256 answer tokens"], ["Reference models", "Majority/chance control, TF-IDF logistic regression, style-feature logistic regression"]], [175,341])
    report.heading("Metrics", 2)
    report.paragraph("Accuracy is the fraction of correctly classified response rows. Precision and recall are calculated separately for each family; their harmonic mean is F1. Macro-F1 averages those family F1 values with equal weight. Balanced class counts make accuracy intuitive, while Macro-F1 exposes one-family collapse. Row-normalized confusion matrices reveal which true family is misidentified as which other family. Metrics are reported as fractions in machine-readable files and percentages in this report.")
    report.heading("Two different sources of uncertainty", 2)
    report.paragraph("The main tables show the mean and sample standard deviation across three model seeds, in percentage points. This describes optimization variation. Saved bootstrap intervals resample near-duplicate template components, carrying every question and its three answer rows together. They estimate held-out-sample uncertainty conditional on the fitted models. Resampling individual answer rows would ignore their shared-question/template dependence. Paired comparisons use the same resampled clusters for both conditions and average metric differences across matched model seeds. Exploratory intervals are not adjusted for multiple comparisons, and no confirmatory significance claim is made.")
    report.heading("Recorded execution evidence", 2)
    report.table(["Response-only run", "Selected epoch", "Run seconds", "Vocabulary", "Parameters"], [[MODEL_NAMES[m]+" / seed 42", str(reps[m]["best_epoch"]), f"{reps[m]['seconds']:.1f}", str(reps[m]["vocab_size"]),f"{reps[m]['parameter_count']:,}"] for m in ("cnn","lstm")], [165,87,87,87,90])
    environment = reference.get("environment",{})
    report.paragraph(f"Execution used {environment.get('device','CPU')} with {config.get('threads',2)} PyTorch computation threads, Python {environment.get('python','see artifacts')}, PyTorch {environment.get('torch','see artifacts')} and NumPy {environment.get('numpy','see artifacts')}. Run seconds include training, selected-model evaluation and artifact work; they are descriptive hardware-dependent measurements.","small")
    report.paragraph("Raw histories, predictions, configurations and run-specific metrics are stored under artifacts/runs/. Checkpoints and vocabularies for the saved seed-42 neural runs permit inference without retraining. Environment and dataset hashes are recorded alongside the results; requirements-lock.txt preserves installed package versions.")
    report.paragraph("No performance target or guaranteed grade is inferred from the assignment. The evaluation emphasizes justified design, correct implementations, transparent measurements, analysis and lessons.", "small")

    # Page 7: main findings.
    report.page("6. RQ1 and RQ2: required experiment results")
    report.table(["Input", "Model", "Seeds", "Accuracy: mean +/- SD", "Macro-F1: mean +/- SD"], [[MODE_NAMES[s["mode"]], MODEL_NAMES[s["model"]], str(s["n"]), metric_mean_std(s,"accuracy"), metric_mean_std(s,"macro_f1")] for s in neural_summary], [119,61,42,146,148])
    report.figure(figures["main"], "Figure 1. Means over seeds 42/43/44 on one fixed test set. Error bars are sample SD, not a confidence interval.")
    report.paragraph(f"RQ1: response-only classification reaches {fmt(out['cnn']['accuracy'])} for CNN and {fmt(out['lstm']['accuracy'])} for BiLSTM. These are measured attribution results on the curated historical-model benchmark. RQ2: the combined condition changes CNN accuracy by {(combined['cnn']['accuracy']-out['cnn']['accuracy'])*100:+.1f} pp and BiLSTM accuracy by {(combined['lstm']['accuracy']-out['lstm']['accuracy'])*100:+.1f} pp. Prompt-only classification remains the structural chance control. Response evidence is equally capped at 256 tokens in both conditions, so a combined-score change is not caused by sacrificing response tokens to the question.")
    cnn_ci = next(c for c in stat["comparisons"] if c["comparison"] == "cnn_input_output_minus_output")["metrics"]["accuracy"]["ci95"]
    lstm_ci = next(c for c in stat["comparisons"] if c["comparison"] == "lstm_input_output_minus_output")["metrics"]["accuracy"]["ci95"]
    report.paragraph(comparison_summary("cnn_input_output_minus_output", "CNN context change") + " " + comparison_summary("lstm_input_output_minus_output", "BiLSTM context change") + " CNN pooling can retain local response patterns; the BiLSTM reverse final state summarizes toward the question prefix in combined input. Longer sequences, changed vocabulary and optimization are plausible contributors, not isolated mechanisms. Attention or response-only pooling would be useful controls.")
    report.paragraph("TF-IDF and style-feature references are detailed in the appendix. They use the same split but full text rather than neural token limits, making them informative practical references rather than perfectly capacity-matched architecture controls.", "small")

    # Page 8: errors + history.
    report.page("7. Error structure and learning behavior")
    report.figure(figures["confusion"], "Figure 2. Response-only seed 42, selected in advance for inspection. Cells show row counts and true-class percentages; rows are actual labels, columns are predictions.")
    for model in ("cnn","lstm"):
        r = reps[model]
        cm = np.array(r["confusion_matrix"])
        off = cm.copy()
        np.fill_diagonal(off, -1)
        i,j = np.unravel_index(off.argmax(),off.shape)
        recall = cm.diagonal()/cm.sum(axis=1)
        weakest = int(recall.argmin())
        report.paragraph(f"For {MODEL_NAMES[model]}, the most frequent directional confusion is {r['labels'][i]} -> {r['labels'][j]} ({cm[i,j]} responses). The lowest class recall is {r['labels'][weakest]} at {fmt(recall[weakest])}. These are error descriptions for the prespecified representative run, not a cherry-picked best seed.")
    report.figure(figures["learning"], "Figure 3. Actual train/validation losses for response-only seed 42. Dashed lines indicate the epoch selected by validation Macro-F1; loss itself is not the checkpoint-selection criterion.")
    report.paragraph("Training loss can decrease while validation loss rises; that pattern is compatible with overfitting or changing calibration, and motivates validation-based selection. The selected checkpoint is not necessarily the minimum validation-loss epoch because the declared criterion is Macro-F1. Three seeds and a separate test partition help reveal variability but do not replace larger or external evaluations.", "small")

    # Page 9: source holdout.
    report.page("8. RQ3: generalization across question sources")
    domain_runs = [r for r in runs if r["experiment"] == "domain_holdout" and r.get("heldout_source")]
    if not domain_runs:
        raise ValueError("RQ3 completed domain-holdout experiment is required")
    heldouts = sorted({holdout_name(r) for r in domain_runs})
    report.paragraph("The transfer experiment holds out the source/domain " + ", ".join(heldouts) + " from training and validation. All three family responses to each held-out question form the test set. A new vocabulary and new classifier are fitted to the remaining sources. This tests a concrete source distribution shift. Upstream source names are not guaranteed to map cleanly to semantic tasks such as all coding or all writing; the result is therefore reported as source generalization rather than universal cross-task robustness.")
    report.table(["Held-out source", "Model / input", "Train / val / test rows", "Accuracy", "Macro-F1"], [[holdout_name(r), MODEL_NAMES[r["model"]]+" / "+MODE_NAMES[r["mode"]], f"{r['train_rows']} / {r['validation_rows']} / {r['n_test_rows']}", fmt(r["accuracy"]), fmt(r["macro_f1"])] for r in domain_runs], [91,156,121,74,74])
    if "domain" in figures:
        report.figure(figures["domain"], "Figure 4. Main response-only accuracy averages three seeds; source-holdout runs use seed 42. Test questions and training counts differ, so this is descriptive comparison rather than a paired causal estimate.")
    for r in domain_runs:
        if r["model"] in ("cnn","lstm") and r["mode"] == "output":
            delta = r["accuracy"]-out[r["model"]]["accuracy"]
            report.paragraph(f"{MODEL_NAMES[r['model']]} on {holdout_name(r)} differs by {delta*100:+.1f} percentage points from its mixed-source main mean. The held-out-source accuracy is {fmt(r['accuracy'])}; the Macro-F1 is {fmt(r['macro_f1'])}. Differences may arise from unseen question style, topic/format composition and less training data. A smaller drop does not establish identical fingerprints across every domain.")
    report.paragraph("A single seed and one held-out source limit precision. An expanded study should repeat leave-one-source-out evaluation with multiple seeds, train-size-matched controls and semantically annotated task domains. The present experiment gives a reproducible, bounded answer to RQ3.")

    # Additional substantive RQ3 page: distinct semantic holdout, not source relabeling.
    writing_runs = [r for r in runs if r["experiment"] == "domain_holdout" and r.get("heldout_domain") == "writing"]
    if len(writing_runs) != 5:
        raise ValueError("The five completed secondary writing-holdout runs are required")
    writing_reference = next(r for r in writing_runs if r["model"] == "cnn")
    report.page("8b. RQ3: transfer to writing prompts")
    report.paragraph("A second transfer study asks whether models trained on non-writing questions can identify the families on writing-like requests. Fixed ordered prompt-only keyword rules define the writing tag (for example poem, story, essay, rewrite, summarize, translate and letter). Programming, mathematics and science tags have priority in the rule order, and broad terms such as write can cause false positives. This is a reproducible approximate semantic split, not an expert-annotated clean genre partition.")
    report.table(["Protocol", "Writing holdout"], [["Test questions / response rows", f"{writing_reference['n_test_rows']//3} / {writing_reference['n_test_rows']}; all three families matched per question"], ["Training / validation questions", f"{writing_reference['train_rows']//3} / {writing_reference['validation_rows']//3}; no writing test component used in fitting"], ["Model / token-budget settings", "Same main configuration; seed 42; newly fitted vocabularies and models"], ["Selection", "Validation Macro-F1 on non-writing data; no writing-test tuning"]],[170,346])
    report.table(["Model / input", "Accuracy", "Macro-F1"], [[MODEL_NAMES[r["model"]]+" / "+MODE_NAMES[r["mode"]],fmt(r["accuracy"]),fmt(r["macro_f1"])] for r in writing_runs],[304,106,106])
    report.figure(figures["writing_confusion"],"Figure 5. Seed-42 neural response-only predictions on writing-tagged questions. Counts and row percentages expose family-specific transfer errors.")
    writing_neural = [r for r in writing_runs if r["model"] in ("cnn","lstm")]
    comparisons_text = []
    for r in writing_neural:
        cm = np.array(r["confusion_matrix"])
        rec = cm.diagonal()/cm.sum(axis=1)
        comparisons_text.append(f"{MODEL_NAMES[r['model']]} changes by {(r['accuracy']-out[r['model']]['accuracy'])*100:+.1f} pp relative to the main mean; its lowest writing-test recall is {r['labels'][int(rec.argmin())]} ({fmt(rec.min())})")
    report.paragraph(". ".join(comparisons_text)+". Test populations and training sizes differ, so these gaps do not isolate task shift as the sole cause.","small")

    # Page 10: feature evidence.
    report.page("9. RQ4: linguistic and structural evidence")
    report.paragraph(f"These statistics use the {len(test_data):,} main test responses, containing the same {test_data.prompt_id.nunique()} questions for each family. Length is word count; TTR is unique lowercased words / total words. Full TTR is length-sensitive, so we also use the first 64 words (or all words for shorter answers). List/code-fence usage is the share of responses containing that format. These full-text descriptors are distinct from the neural token budgets.")
    feature_rows = data_feature_rows(test_data)
    report.table(["Family", "Mean words", "Median words", "Full TTR", "64-word TTR", "List usage", "Code fences"], feature_rows, [75,70,73,65,78,78,77])
    longest = max(feature_rows,key=lambda r:float(r[1]))
    highest_ttr = max(feature_rows,key=lambda r:float(r[4]))
    highest_list = max(feature_rows,key=lambda r:float(r[5].replace("%","")))
    report.paragraph(f"In this test sample, {longest[0]} has the highest average response length ({longest[1]} words), {highest_ttr[0]} has the highest mean 64-word-prefix TTR ({highest_ttr[4]}), and {highest_list[0]} most often uses lists ({highest_list[5]}). Full-response TTR differences partly track length; the prefix comparison reduces but does not eliminate that confound. Code fences are rare, so their rates should not be generalized broadly.","small")
    report.figure(figures["features"], "Figure 6. Test-response word-count distributions (outliers hidden in the plot only) and mean prefix lexical diversity. These describe model-associated features, not causal network attribution.")
    ablations = [r for r in runs if r["experiment"] == "ablation"]
    if not ablations:
        raise ValueError("RQ4 completed ablation experiments are required")
    report.table(["Model", "Response condition (seed 42)", "Accuracy", "Macro-F1"], [[MODEL_NAMES[r["model"]], VARIANT_NAMES.get(r.get("variant","raw"),r.get("variant","raw")), fmt(r["accuracy"]), fmt(r["macro_f1"])] for r in [reps["cnn"],reps["lstm"]]+ablations], [104,246,83,83])
    report.paragraph("Each ablation retrains with seed 42 on the same split and compares against the original seed-42 run. Normalization removes Markdown/list/fence markers and collapses whitespace; it changes several properties together. The 64-token condition caps the observed prefix, while shorter-answer length remains visible. Feature-only classification and these transformations show association and sensitivity, not proof of the neural model's exact causal signals.","small")

    # Page 11: uncertainty/limits/lessons. Exact bootstrap schema adapted later.
    report.page("10. Interpretation, limitations and lessons")
    report.heading("Paired statistical comparisons", 2)
    if stat:
        comparisons = stat.get("comparisons", stat.get("paired_comparisons", [])) if isinstance(stat,dict) else []
        if isinstance(comparisons,dict):
            comparisons = [{"comparison": key, **value} for key,value in comparisons.items()]
        rendered = []
        for c in comparisons:
            name = c.get("comparison",c.get("name",c.get("label","Paired difference")))
            metric = c.get("metrics",{}).get("accuracy",{})
            point = metric.get("delta",c.get("accuracy_delta",c.get("delta_accuracy",c.get("accuracy_difference"))))
            ci = metric.get("ci95",c.get("accuracy_ci95",c.get("delta_accuracy_ci95",c.get("bootstrap_accuracy_ci95"))))
            if point is not None and ci is not None:
                rendered.append([comparison_label(name),f"{float(point)*100:+.1f} pp","["+", ".join(f"{float(v)*100:+.1f}" for v in ci)+"] pp"])
        if rendered:
            report.table(["Paired contrast", "Accuracy difference", "95% group-bootstrap interval"], rendered,[241,116,159])
            report.paragraph(f"Intervals use {stat['repetitions']:,} paired bootstrap resamples of template clusters (seed {stat['seed']}). " + comparison_summary("cnn_minus_lstm_output", "Response-only CNN minus BiLSTM") + " " + comparison_summary("cnn_normalized_minus_raw", "CNN normalization change") + " " + comparison_summary("lstm_normalized_minus_raw", "BiLSTM normalization change") + " Intervals are exploratory and unadjusted for multiple comparisons.","small")
        else:
            report.paragraph("Prompt-cluster paired bootstrap results are saved in artifacts/statistical_analysis.json. Point estimates and interval calculations use the recorded predictions. The file, rather than seed standard deviations, is the evidence for sampling uncertainty.")
    else:
        report.paragraph("Run-specific 95% accuracy intervals are saved in metrics.json and resample question groups. Seed standard deviations alone must not be read as a significance test for differences between conditions.")
    report.heading("What the data supports and what it cannot establish", 2)
    report.paragraph("The controlled prompt-only experiment separates question imbalance from response attribution. Response scores show that information associated with the three historical generators is recoverable in this curated benchmark. They do not separate intrinsic model style from system prompts, generation caps, preference-alignment choices, response correctness or refusal behavior. Only one version of each family is represented. The sample is modest, English-heavy and noncommercial benchmark data; similar questions and source bias may remain.")
    report.paragraph("The task excludes human-written text, unregistered families, mixed-model documents, translated or edited responses and adversarial paraphrasing. A classifier must output one of its three known labels even when the real author is absent. These results should not be used as conclusive evidence about a person's writing or a current deployed model. Current-version collection, explicit unknown-class handling and independent external data would be necessary before any broader application.")
    report.page("10b. Lessons and experience from the project")
    report.heading("1. Source review changed the final experiment", 2)
    report.paragraph(f"Reviewing the Koala README exposed a request against training use that a general benchmark license notice did not resolve. The earlier candidate subset included Koala examples. The final protocol excludes all {dropped['excluded_koala_training_request']} original Koala prompt groups before any split or fitting, leaving {prompt_count} matched questions. All 38 experiments were rerun, including the holdouts and ablations; earlier scores are not presented as measurements of the revised data. The verifier rejects either Koala or OASST in the submitted dataset. The lesson is to check constituent-source requests before training and to regenerate dependent results when curation changes.")
    report.heading("2. A low control score can be the correct result", 2)
    report.paragraph("Prompt-only accuracy is exactly 33.3% across the required neural runs because each identical question occurs under all three labels. Instead of treating this as a failed classifier, the project uses it as a negative control. Grouping questions and near-duplicate templates before splitting also prevents related examples from crossing partitions. The lesson is to derive the expected control result from data construction before interpreting model accuracy; a higher prompt-only score here would require investigating imbalance or leakage.")
    report.heading("3. Context comparisons need an equal response budget", 2)
    report.paragraph("A single total length cap would allow a prepended question to displace answer tokens, mixing the effect of context with lost response evidence. The implemented comparison retains 256 answer tokens in both conditions and adds up to 64 question tokens plus a separator for combined input. A regression test checks the shared answer budget even for a 300-token question. CNN masks padded convolution windows and BiLSTM packs sequence lengths; padding-invariance tests check that adding padded suffixes does not change predictions. The lesson is to verify input construction and batching behavior before explaining architecture differences.")
    report.heading("4. Observed differences need uncertainty and bounded explanations", 2)
    report.paragraph(comparison_summary("cnn_input_output_minus_output", "Final CNN context comparison") + " " + comparison_summary("lstm_input_output_minus_output", "Final BiLSTM context comparison") + " Three seeds describe initialization variability on one split. Paired bootstrap instead resamples whole test template groups. Source/writing holdouts use different questions and training sizes, and normalization changes multiple properties. The lesson is to distinguish a measured difference from its cause and to keep seed variability, test-sample uncertainty and domain shift separate.")
    report.heading("5. Reproducibility requires checking artifacts, not only a table", 2)
    report.paragraph("The project retains original-source hashes, curated-data hashes, split membership, training histories, selected checkpoints and row-level prediction probabilities. Verification recomputes metrics for every run and replays all test rows for the 14 saved seed-42 neural checkpoints. These checks connect the manuscript to actual inference and detect stale data or model artifacts. The lesson is to treat a source-exclusion fix as a pipeline change: data, fitting, figures, interpretations, slides and the submission archive must agree.")
    report.heading("AI assistance and responsibility", 2)
    report.paragraph(metadata["ai_assistance"])
    report.paragraph("The report records empirical design lessons rather than inventing individual student experiences or a contribution division. Both named students should review the implementation and measured outputs, correct any remaining issue and be prepared to explain the work before submission.", "small")

    # Appendix: reference models and lexical associations add interpretable evidence.
    report.page("11. Reference classifiers and lexical associations")
    baseline_runs = [r for r in main if r["model"] not in ("cnn","lstm")]
    report.table(["Reference / input", "Selected C", "Accuracy", "Macro-F1"], [[MODEL_NAMES[r["model"]]+" / "+MODE_NAMES[r["mode"]],str(r.get("selected_C") or "n/a"),fmt(r["accuracy"]),fmt(r["macro_f1"])] for r in baseline_runs],[274,76,83,83])
    report.paragraph("The majority baseline always selects the same training-majority family (ties resolve by fixed label order). Uniform-random accuracy is one reproducible seed-42 realization around the 1/3 expectation, so its observed value need not equal exactly 33.3%. TF-IDF extracts unigrams and bigrams, uses min_df 2, sublinear term frequency and at most 40,000 features. Logistic regularization C is chosen from 0.1, 1 and 10 by validation Macro-F1. These references use complete texts, unlike capped neural inputs.")
    lexical = (stat or {}).get("lexical_signals",{})
    if lexical:
        report.heading("Training-fit lexical evidence",2)
        report.table(["Family", "Largest positive TF-IDF coefficients"], [[label,"; ".join(f"{term['ngram']} ({term['coefficient']:+.2f})" for term in terms[:6])] for label,terms in lexical.items()],[91,425])
        report.paragraph("The displayed n-grams come from the response-only TF-IDF classifier fitted on training data, after regularization selection on validation. Positive coefficients support that label relative to alternatives in this fitted classifier. Common lexical cues may reflect refusals, answer structure or dataset composition; correlated features and coefficient scale prevent treating these numbers as intrinsic provider traits or CNN/LSTM explanations.")
    feature_run = next(r for r in main if r["model"] == "features")
    coeff_path = ROOT / "artifacts" / "runs" / feature_run["run_id"] / "feature_coefficients.csv"
    if coeff_path.exists():
        coeff = pd.read_csv(coeff_path)
        rows = []
        for label in feature_run["labels"]:
            ordered = coeff.sort_values(label,ascending=False).head(3)
            rows.append([label,"; ".join(f"{row.feature} ({getattr(row,label):+.2f})" for row in ordered.itertuples(index=False))])
        report.heading("Measurable style-feature reference",2)
        report.table(["Family", "Largest positive standardized feature coefficients"],rows,[91,425])
        report.paragraph("Sixteen handcrafted statistics cover log length/lines, lexical diversity, mean word length, newline/punctuation/digit rates, list-line share and code-fence count. StandardScaler is fit on training rows; coefficients are from its logistic classifier. Full-response TTR depends on length and features are correlated. The classifier's actual test score demonstrates predictive association of this feature set without establishing a causal neural mechanism.","small")

    # Final page: reproducibility and primary references.
    report.page("12. Reproducibility and references")
    report.heading("Artifact map", 2)
    report.table(["Artifact", "Purpose"], [[str(dataset_path.relative_to(ROOT)).replace("\\","/"), "Curated balanced triples with source/domain labels"], ["data/source_manifest.json and data/licenses/", "Source URLs, checksums, filtering and applicable notices"], ["src/llm_attribution/", "Tokenizer, group splits, CNN, BiLSTM, training and inference"], ["artifacts/results.csv and results.json", "All completed experiment metrics, seed and condition labels"], ["artifacts/runs/<run_id>/", "Predictions, confusion/per-class metrics, histories, checkpoint and vocabulary where saved"], ["artifacts/statistical_analysis.json", "Paired template-cluster uncertainty estimates"], ["report/project_report.pdf / .md", "Submission report and matching readable source"], ["docs/DEFENSE_GUIDE_KO.md", "Korean explanation and presentation preparation"]], [258,258])
    report.heading("Reproduction", 2)
    report.paragraph("Use Python 3.12 and install requirements-lock.txt with the CPU wheel index specified in README.md. Run scripts/prepare_data.py, scripts/run_experiments.py, scripts/run_additional_domains.py (writing holdout), scripts/analyze_results.py and scripts/build_report.py, in that order. Build slides with scripts/build_slides.py and validate with scripts/verify_submission.py. README.md gives exact commands. Fixed splits and seeds are preserved; hardware/library changes can alter floating-point behavior.")
    report.heading("Primary references", 2)
    references = [
        "[1] CMPSC 448. Who Wrote It? Identifying LLMs from Their Responses. Assignment, modified September 17, 2026. Supplied course PDF.",
        "[2] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. (2023). AlpacaEval: An Automatic Evaluator of Instruction-following Models. GitHub repository. https://github.com/tatsu-lab/alpaca_eval . Model-output data reuse; no AlpacaEval preference evaluation is performed. Revisions are pinned in data/source_manifest.json.",
        "[3] Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP, 1746-1751. https://aclanthology.org/D14-1181/ . DOI: 10.3115/v1/D14-1181.",
        "[4] Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735 .",
        "[5] PyTorch 2.6 documentation. Conv1d, LSTM and pack_padded_sequence. https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv1d.html ; https://docs.pytorch.org/docs/2.6/generated/torch.nn.LSTM.html ; https://docs.pytorch.org/docs/2.6/generated/torch.nn.utils.rnn.pack_padded_sequence.html .",
        "[6] Creative Commons. Attribution-NonCommercial 4.0 International. https://creativecommons.org/licenses/by-nc/4.0/ . Dataset notice preserved in data/licenses/.",
    ]
    for text in references:
        report.paragraph(text,"small")
    report.paragraph("Submission repository: " + str(metadata.get("repository_url") or "see README.md") + ". The repository must contain the source code and this PDF. Uploading files to GitHub and submitting the repository URL through the course channel are separate actions.","small")
    report.page("Dataset-source references and reuse scope")
    report.paragraph("The retained question sources are credited below, following AlpacaEval's evaluation-set citation guidance. Koala is cited only to explain its exclusion from the final dataset. These references describe upstream benchmarks; their published scores are not experimental results of this project.")
    source_references = [
        "[7] Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. ACL (Volume 1: Long Papers), 13484-13508. https://aclanthology.org/2023.acl-long.754/ . DOI: 10.18653/v1/2023.acl-long.754.",
        "[8] Chiang, W.-L., et al. (2023, March 30). Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. LMSYS blog. https://lmsys.org/blog/2023-03-30-vicuna/ . The percentage belongs to the original blog title and is not an attribution accuracy claim.",
        "[9] Gudibande, A., and contributors. Koala Evaluation Set. GitHub repository. https://github.com/arnav-gudibande/koala-test-set . Excluded source: its README requests no training use. No publication year or paper DOI is asserted for this repository.",
        "[10] Anthropic. HH-RLHF dataset card and repository (helpful-base subset). https://huggingface.co/datasets/Anthropic/hh-rlhf ; https://github.com/anthropics/hh-rlhf . This project uses AlpacaEval's published questions and generator responses, not HH-RLHF preference labels.",
    ]
    for text in source_references:
        report.paragraph(text,"small")
    report.heading("Koala exclusion and reuse scope",2)
    report.paragraph("Koala's README asks that its evaluation set not be used for training. All Koala-derived prompt groups are excluded from the final dataset before splitting, so none are used in training, validation or testing. The 38 final runs were retrained on the revised subset; the earlier Koala-inclusive candidate measurements are superseded. The source request and exclusion are preserved in data/NOTICE.md and the curation audit. CC BY-NC 4.0 attribution and noncommercial-use requirements still apply to the retained AlpacaEval data.","small")
    report.paragraph("Reference metadata, immutable output-file checksums, API documentation checks and factual corrections are documented in docs/FACT_CHECK_KO.md and artifacts/reference_source_checks.json. Web access date: October 11, 2026.","small")
    path = report.build()
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    start,end = "<!-- MEASURED_RESULTS_START -->","<!-- MEASURED_RESULTS_END -->"
    measured = ["### Measured main results", "", f"The same {reference['n_test_rows']} held-out response rows are used for all main conditions. Values are means over three seeds; sample SD is in percentage points.", "", "| Input | Model | Accuracy | Macro-F1 |", "| --- | --- | --- | --- |"]
    measured.extend(f"| {MODE_NAMES[s['mode']]} | {MODEL_NAMES[s['model']]} | {metric_mean_std(s,'accuracy')} | {metric_mean_std(s,'macro_f1')} |" for s in neural_summary)
    measured.extend(["", comparison_summary("cnn_input_output_minus_output", "CNN context change") + " " + comparison_summary("lstm_input_output_minus_output", "BiLSTM context change") + " Source and writing holdouts show the limits of distribution transfer. Full results and intervals are in the PDF and saved JSON.", ""])
    if start not in readme or end not in readme:
        raise ValueError("README measured-result markers are required")
    readme_path.write_text(readme.split(start)[0]+start+"\n\n"+"\n".join(measured)+"\n"+end+readme.split(end)[1],encoding="utf-8")
    feature_csv = ROOT / "report" / "test_feature_summary.csv"
    with feature_csv.open("w",encoding="utf-8",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family","mean_words","median_words","full_ttr","prefix64_ttr","list_usage","code_fence_usage"])
        writer.writerows(data_feature_rows(test_data))
    print(f"Generated {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_report()

"""Build an optional eight-slide talk from completed, measured project artifacts.

The English PDF and Korean speaker notes are generated together. No experiment
metric is hand-entered; missing or incomplete artifacts stop the build.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 540
NAVY = "#10283F"
TEAL = "#0C8590"
BLUE = "#3F70B5"
ORANGE = "#D78340"
INK = "#1F3548"
GRAY = "#5B6D7B"
LIGHT = "#F2F6F8"
PALE = "#DFEEF0"
WHITE = "#FFFFFF"
MODEL = {"cnn": "TextCNN", "lstm": "BiLSTM", "tfidf": "TF-IDF + LR"}
MODE = {"input": "Prompt only", "output": "Response only", "input_output": "Prompt + response"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def ci(interval: list[float]) -> str:
    return "[" + ", ".join(f"{100 * value:+.1f}" for value in interval) + "] pp"


def register_fonts() -> None:
    normal = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = Path("C:/Windows/Fonts/segoeuib.ttf")
    if normal.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("Talk", str(normal)))
        pdfmetrics.registerFont(TTFont("TalkBold", str(bold)))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font("Talk", "Helvetica", "WinAnsiEncoding"))
        pdfmetrics.registerFont(pdfmetrics.Font("TalkBold", "Helvetica-Bold", "WinAnsiEncoding"))
    pdfmetrics.registerFontFamily("Talk", normal="Talk", bold="TalkBold")


class Deck:
    def __init__(self, destination: Path, metadata: dict):
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = canvas.Canvas(str(destination), pagesize=(WIDTH, HEIGHT), pageCompression=1)
        self.pdf.setTitle("Who Wrote It? — CMPSC 448 Project Presentation")
        self.pdf.setAuthor(", ".join(metadata["authors"]))
        self.pdf.setSubject("Eight-slide presentation of measured CNN and BiLSTM attribution experiments")
        self.metadata = metadata
        self.number = 0
        self.dark = False

    def box(self, x, y, width, height, fill=LIGHT, radius=10):
        self.pdf.setFillColor(colors.HexColor(fill))
        self.pdf.roundRect(x, y, width, height, radius, fill=1, stroke=0)

    def text(self, x, top, width, text, size=18, color=INK, bold=False, leading=None):
        style = ParagraphStyle("talk", fontName="TalkBold" if bold else "Talk",
                               fontSize=size, leading=leading or size * 1.3,
                               textColor=colors.HexColor(color))
        paragraph = Paragraph(text, style)
        _, height = paragraph.wrap(width, HEIGHT)
        if top - height < (10 if top < 40 else 38):
            raise ValueError(f"Slide {self.number} text exceeds its content area: {text[:80]}")
        paragraph.drawOn(self.pdf, x, top - height)
        return height

    def start(self, title: str, section: str, dark=False):
        if self.number:
            self.pdf.showPage()
        self.number += 1
        self.dark = dark
        self.pdf.setFillColor(colors.HexColor(NAVY if dark else WHITE))
        self.pdf.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        self.pdf.setFillColor(colors.HexColor(TEAL))
        self.pdf.rect(0, HEIGHT - 7, WIDTH, 7, fill=1, stroke=0)
        self.text(48, 506, 860, section.upper(), 11, PALE if dark else TEAL, True)
        self.text(48, 478, 865, title, 29, WHITE if dark else NAVY, True, 35)
        self.pdf.setStrokeColor(colors.HexColor("#365064" if dark else "#D8E2E8"))
        self.pdf.line(48, 40, WIDTH - 48, 40)
        footer = f"CMPSC 448  |  {' & '.join(self.metadata['authors'])}"
        self.text(48, 28, 800, html.escape(footer), 9, PALE if dark else GRAY)
        self.text(884, 28, 35, f"{self.number:02d} / 08", 9, PALE if dark else GRAY)

    def table(self, x, top, widths, rows, size=16, row_height=44):
        wrapped = []
        for index, row in enumerate(rows):
            wrapped.append([Paragraph(html.escape(str(cell)), ParagraphStyle(
                "cell", fontName="TalkBold" if index == 0 else "Talk", fontSize=size,
                leading=size * 1.2, textColor=colors.HexColor(WHITE if index == 0 else INK)))
                            for cell in row])
        table = Table(wrapped, colWidths=widths, rowHeights=[row_height] * len(rows))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(LIGHT), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#CBD8E0")),
        ]))
        _, height = table.wrap(sum(widths), HEIGHT)
        if top - height < 48:
            raise ValueError(f"Slide {self.number} table exceeds its content area")
        table.drawOn(self.pdf, x, top - height)
        return height

    def diagram(self, x, top, width, stages, accent=TEAL):
        height = 42
        for index, stage in enumerate(stages):
            y = top - index * 61 - height
            self.box(x, y, width, height, PALE if index in (0, len(stages)-1) else LIGHT, 7)
            self.text(x + 14, y + 31, width - 28, stage, 16, INK, index in (0, len(stages)-1))
            if index < len(stages)-1:
                self.pdf.setStrokeColor(colors.HexColor(accent))
                self.pdf.setLineWidth(1.5)
                center = x + width / 2
                self.pdf.line(center, y - 2, center, y - 16)
                self.pdf.line(center, y - 16, center - 4, y - 11)
                self.pdf.line(center, y - 16, center + 4, y - 11)

    def image(self, path, x, y, width, height):
        if not path.exists():
            raise FileNotFoundError(path)
        self.pdf.drawImage(str(path), x, y, width=width, height=height,
                           preserveAspectRatio=True, anchor="c", mask="auto")

    def save(self):
        if self.number != 8:
            raise ValueError("The presentation must contain exactly eight slides")
        self.pdf.save()


def main_mean(runs, model, mode, metric="accuracy"):
    selected = [r for r in runs if r["experiment"] == "main" and r["variant"] == "raw"
                and r["model"] == model and r["mode"] == mode]
    if not selected:
        raise ValueError(f"Missing main results: {model}/{mode}")
    return statistics.mean(r[metric] for r in selected), (
        statistics.stdev(r[metric] for r in selected) if len(selected) > 1 else 0)


def unique_run(runs, **conditions):
    found = [r for r in runs if all(r.get(key) == value for key, value in conditions.items())]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one run for {conditions}, got {len(found)}")
    return found[0]


def build(root: Path) -> tuple[Path, Path]:
    register_fonts()
    metadata = read(root / "metadata.json")
    results = read(root / "artifacts/results.json")
    stats = read(root / "artifacts/statistical_analysis.json")
    runs = results["runs"]
    if len(runs) != 38:
        raise ValueError(f"Final presentation expects 38 completed runs, found {len(runs)}")
    data = results["metadata"]["dataset"]
    config = results["metadata"]["training_config"]
    split = results["metadata"]["split_counts_prompts"]
    comparisons = {item["comparison"]: item for item in stats["comparisons"]}
    destination = root / "report/presentation.pdf"
    talk = Deck(destination, metadata)

    talk.start("Who wrote it?", "A closed-set LLM attribution experiment", dark=True)
    talk.text(48, 385, 800, "Identifying the source model family<br/>from its generated response", 32, WHITE, True, 41)
    talk.text(48, 272, 800, "TextCNN vs. BiLSTM  ·  matched prompts  ·  measured results", 19, PALE)
    for x, label, color in [(48, "GPT-4-0613", BLUE), (320, "Claude-2", TEAL), (592, "Gemini-Pro", ORANGE)]:
        talk.box(x, 163, 240, 56, color, 9)
        talk.text(x + 18, 204, 212, label, 21, WHITE, True)
    talk.text(48, 128, 830, f"Team leader: {html.escape(metadata['team_leader'])}  |  Team member: "
              + html.escape(", ".join(metadata["team_members"])), 17, WHITE)
    talk.text(48, 89, 830, "Four questions: identification, prompt context, transfer, and stylistic signals.", 15, PALE)

    talk.start("Matched questions make the control meaningful", "Data and leakage prevention")
    cards = [(48, data["prompts"], "shared prompts"), (345, data["rows"], "model responses"),
             (642, data["split_groups"], "template components")]
    for x, value, label in cards:
        talk.box(x, 290, 270, 120)
        talk.text(x + 20, 391, 230, f"{value:,}", 42, TEAL, True)
        talk.text(x + 20, 326, 230, label, 17, GRAY)
    talk.text(48, 263, 860, "One question → three answers. Each family has exactly " + str(data["prompts"]) + " rows.", 21, INK, True)
    talk.table(48, 221, [280, 285, 299], [
        ["Training", "Validation", "Test"],
        [f"{split['train']} prompts / {split['train'] * 3} rows", f"{split['validation']} prompts / {split['validation'] * 3} rows",
         f"{split['test']} prompts / {split['test'] * 3} rows"],
    ], 17, 41)
    talk.text(48, 122, 860, "Near-duplicate templates stay together. Vocabulary is learned on training data only.<br/>"
              "Balanced prompt-only inputs must yield 33.3% accuracy; explicit model self-identities are redacted.", 16, GRAY)
    talk.text(48, 70, 850, "Source: official AlpacaEval outputs; exclusions, licenses, model IDs and checksums are documented.", 12, GRAY)

    talk.start("Two small classifiers, one evaluation protocol", "Models and training")
    talk.text(48, 414, 415, "TextCNN", 23, TEAL, True)
    talk.text(500, 414, 415, "Bidirectional LSTM", 23, BLUE, True)
    talk.diagram(48, 373, 408, ["Token IDs → 64-dim embeddings", "Kernel sizes 3 / 4 / 5; 48 filters each",
                                "Padding-aware max pooling", "Dropout 0.3 → 3-family softmax"], TEAL)
    talk.diagram(500, 373, 408, ["Token IDs → 64-dim embeddings", "48 hidden units in each direction",
                                 "Packed sequences → final hidden states", "Dropout 0.3 → 3-family softmax"], BLUE)
    talk.text(48, 118, 865, "Fair answer budget: response ≤256 tokens; prompt ≤64; combined ≤321 with separator.<br/>"
              f"Seeds 42 / 43 / 44; AdamW lr {config['learning_rate']}; batch {config['batch_size']}; "
              f"≤{config['epochs']} epochs; stop after {config['patience']} stale epochs.", 16, GRAY)
    talk.text(48, 65, 865, "Select by validation Macro-F1, then validation loss. Evaluate the restored checkpoint once on the test set.", 12, GRAY)

    talk.start("RQ1: response text identifies source families above chance", "Required six conditions")
    main_rows = [["Architecture / input", "Accuracy", "Macro-F1"]]
    for model in ("cnn", "lstm"):
        for mode in ("input", "output", "input_output"):
            accuracy, sd = main_mean(runs, model, mode)
            f1, f1sd = main_mean(runs, model, mode, "macro_f1")
            main_rows.append([f"{MODEL[model]} · {MODE[mode]}", f"{pct(accuracy)} ± {100*sd:.1f} pp",
                              f"{pct(f1)} ± {100*f1sd:.1f} pp"])
    talk.table(48, 408, [386, 236, 242], main_rows, 16, 40)
    tfidf = unique_run(runs, experiment="main", model="tfidf", mode="output")
    talk.text(48, 106, 860, f"Mean ± sample SD over three training seeds. Response-only TF-IDF baseline: {pct(tfidf['accuracy'])}.<br/>"
              "All prompt-only neural controls equal the theoretical 33.3% chance accuracy.", 17, GRAY)

    talk.start("RQ2: does prompt context help?", "Paired test-cluster comparisons")
    for x, model, color in [(48, "cnn", TEAL), (500, "lstm", BLUE)]:
        result = comparisons[f"{model}_input_output_minus_output"]["metrics"]["accuracy"]
        talk.box(x, 226, 410, 185)
        talk.text(x + 20, 389, 370, MODEL[model], 22, color, True)
        talk.text(x + 20, 348, 370, pp(result["delta"]), 43, color, True)
        talk.text(x + 20, 285, 370, "Combined − response-only accuracy", 17, INK)
        talk.text(x + 20, 256, 370, "Paired 95% CI: " + ci(result["ci95"]), 15, GRAY)
    cnn_interval = comparisons["cnn_input_output_minus_output"]["metrics"]["accuracy"]["ci95"]
    uncertainty = "CNN's interval includes zero; its observed gain is inconclusive." if cnn_interval[0] <= 0 <= cnn_interval[1] else "CNN's paired interval excludes zero in this fixed test sample."
    talk.text(48, 197, 858, uncertainty + "<br/>Same questions, same seeds, same response budget for both architectures.", 19, INK)
    talk.text(48, 130, 858, "Hypothesis: prepended prompts alter final reverse-LSTM states and optimization over longer sequences.<br/>"
              "This is not a causal proof; attention or separate encoders would test it.", 15, GRAY)
    talk.text(48, 82, 858, f"{stats['repetitions']:,} paired bootstrap resamples of complete template components; "
              "test-sample uncertainty only, with no multiple-comparison claim.", 13, GRAY)

    talk.start("RQ3: evaluate both source shift and writing tasks", "Transfer beyond the mixed training split")
    transfer_rows = [["Response-only model", "Main test", "SelfInstruct held out", "Writing held out"]]
    for model in ("cnn", "lstm", "tfidf"):
        source = unique_run(runs, experiment="domain_holdout", model=model, mode="output", heldout_source="selfinstruct")
        writing = unique_run(runs, experiment="domain_holdout", model=model, mode="output", heldout_domain="writing")
        main_acc, _ = main_mean(runs, model, "output")
        transfer_rows.append([MODEL[model], pct(main_acc), pct(source["accuracy"]), pct(writing["accuracy"])])
    talk.table(48, 407, [242, 173, 231, 218], transfer_rows, 17, 48)
    source = unique_run(runs, experiment="domain_holdout", model="cnn", mode="output", heldout_source="selfinstruct")
    writing = unique_run(runs, experiment="domain_holdout", model="cnn", mode="output", heldout_domain="writing")
    talk.text(48, 193, 860, f"SelfInstruct: verified upstream source; {source['n_test_rows']//3} unseen prompts.<br/>"
              f"Writing: deterministic input-only category; {writing['n_test_rows']//3} unseen prompts.", 19, INK, True)
    talk.text(48, 125, 860, "Holdouts are exploratory seed42 runs. Main neural results average three seeds.<br/>"
              "Training size and test distributions change; differences do not isolate a causal domain effect.", 16, GRAY)
    talk.text(48, 72, 860, "Template components cannot cross train/validation/test; non-target component members are purged when necessary.", 12, GRAY)

    talk.start("RQ4: style and vocabulary carry measurable signals", "Feature baseline and separate ablations")
    style = unique_run(runs, experiment="main", model="features", mode="output")
    talk.box(48, 260, 254, 150, PALE)
    talk.text(67, 391, 216, "Style features only", 19, TEAL, True)
    talk.text(67, 353, 216, pct(style["accuracy"]), 36, TEAL, True)
    talk.text(67, 307, 216, "Length, diversity, punctuation<br/>and formatting (16 features)", 13, GRAY)
    ablation_rows = [["Seed42 output", "TextCNN", "BiLSTM"]]
    for title, experiment, variant in [("Raw / 256 tokens", "main", "raw"),
                                        ("First 64 tokens", "ablation", "length64"),
                                        ("Formatting normalized", "ablation", "normalized")]:
        values = [unique_run(runs, experiment=experiment, model=model, mode="output", variant=variant, seed=42)
                  for model in ("cnn", "lstm")]
        ablation_rows.append([title, pct(values[0]["accuracy"]), pct(values[1]["accuracy"])])
    talk.table(326, 410, [280, 147, 159], ablation_rows, 15, 42)
    talk.text(48, 218, 860, "Training-only lexical associations from TF-IDF", 19, INK, True)
    for index, label in enumerate(data["labels"]):
        terms = [item["ngram"] for item in stats["lexical_signals"][label][:4]]
        x = 48 + index * 297
        talk.text(x, 178, 270, html.escape(label), 16, TEAL, True)
        talk.text(x, 148, 266, html.escape(" / ".join(terms)), 15, INK)
    talk.text(48, 95, 860, "Each condition applies one transformation and retrains; multiple signals can change.<br/>"
              "Lexical terms explain the baseline's associations, not CNN/LSTM decisions or universal provider traits.", 15, GRAY)

    talk.start("What we learned — and what we can claim", "Scope and reproducibility", dark=True)
    takeaways = [
        "Response text predicts these three model families above chance.",
        f"Prompt context changes CNN by {pp(comparisons['cnn_input_output_minus_output']['metrics']['accuracy']['delta'])} and BiLSTM by {pp(comparisons['lstm_input_output_minus_output']['metrics']['accuracy']['delta'])}.",
        "Transfer and style controls reveal useful signals alongside dataset confounds.",
    ]
    top = 406
    for index, takeaway in enumerate(takeaways):
        talk.text(48, top - index * 57, 860, f"{index + 1}.  {takeaway}", 21, WHITE, True)
    talk.box(48, 87, 864, 131, "#203B50")
    talk.text(68, 199, 824, "Limits: historical model versions; one benchmark; unequal source decoding caps;<br/>"
              "small matched dataset; no unknown-family or new-version detection claim.", 17, PALE)
    talk.text(68, 141, 824, "Reproducible package: source + licensed data provenance + 38 measured runs + predictions<br/>"
              "+ checkpoints + report + verification. AI assistance is disclosed; students must understand the work.", 15, PALE)
    talk.text(48, 68, 855, "Source review: Koala excluded from all splits; all 38 runs retrained. See report and data/NOTICE.md.", 12, PALE)
    talk.save()

    notes_path = root / "docs/PRESENTATION_NOTES_KO.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes = korean_notes(runs, stats, metadata, data, split)
    notes += "\n\n## 출처 검토와 최종 재실험\n\nKoala 원문은 평가 데이터의 학습 사용을 피하라고 요청합니다. 최종 데이터는 Koala를 학습·검증·테스트 모두에서 제외하고 38개 실험을 전부 다시 실행했습니다. 제외 근거와 실제 수치는 보고서, `data/NOTICE.md`, `data/processed/audit.json`에서 확인할 수 있습니다. 이전 Koala 포함 결과는 최종 결과가 아닙니다.\n"
    notes_path.write_text(notes, encoding="utf-8")
    return destination, notes_path


def korean_notes(runs, stats, metadata, data, split):
    comparisons = {c["comparison"]: c for c in stats["comparisons"]}
    cnn, _ = main_mean(runs, "cnn", "output")
    lstm, _ = main_mean(runs, "lstm", "output")
    cdelta = comparisons["cnn_input_output_minus_output"]["metrics"]["accuracy"]
    ldelta = comparisons["lstm_input_output_minus_output"]["metrics"]["accuracy"]
    cnn_uncertainty = ("구간에 0이 포함되므로 CNN의 방향성 있는 성능 차이를 확정하지 않습니다."
                       if cdelta['ci95'][0] <= 0 <= cdelta['ci95'][1] else
                       "이 고정 테스트 표본에서 CNN의 구간은 0을 포함하지 않습니다.")
    feature = unique_run(runs, experiment="main", model="features", mode="output")
    leader = metadata["team_leader"]
    member = ", ".join(metadata["team_members"])
    return f"""# 5분 발표용 한국어 메모

`report/presentation.pdf`의 영어 슬라이드 8장에 대응합니다. 숫자는 저장된 최종 실험 결과에서 자동 생성됩니다.

발표 분담 **추천**: {leader}이 1–4장, {member}이 5–8장을 설명하면 약 2분 30초씩입니다. 이것은 발표 역할 제안이며 실제 기여 내역을 주장하지 않습니다. 제출 전 두 사람 모두 모든 실험을 설명할 수 있도록 연습하세요.

## 1장 — 문제와 팀 (25초)

“저희 프로젝트는 답변을 보고 GPT, Claude, Gemini 중 어느 계열이 썼는지 분류합니다. 거대한 LLM을 새로 만드는 것이 아니라, TextCNN과 양방향 LSTM을 학습하고 입력 조건을 비교했습니다. 이 결과는 사용한 세 모델 버전과 데이터에 대한 closed-set 결과입니다.”

## 2장 — 데이터와 누수 방지 (40초)

“공식 AlpacaEval 결과에서 같은 질문에 대한 세 모델의 답변을 짝지었습니다. 총 {data['prompts']}개 질문, {data['rows']}개 답변이고 모델당 {data['prompts']}개라서 정답 분포가 균형적입니다. 비슷한 질문 템플릿을 묶은 {data['split_groups']}개 그룹을 통째로 나눴습니다. 실제 질문 수는 학습 {split['train']}, 검증 {split['validation']}, 테스트 {split['test']}개입니다. 질문만 보는 조건에서는 같은 질문에 세 라벨이 붙으므로 정확도 33.3%가 정상입니다.”

추가 설명: 템플릿 그룹 기준으로 약 65/15/20%를 나누므로 실제 질문 비율은 정확히 같은 숫자가 아닙니다. 모델 자기소개는 동일한 규칙으로 가렸고, 데이터 제외와 출처는 감사 파일에 있습니다.

## 3장 — 모델과 학습 (35초)

“CNN은 3·4·5토큰 크기의 필터와 max pooling을 사용합니다. BiLSTM은 앞뒤 방향의 마지막 은닉 상태를 합칩니다. 패딩이 결과에 영향을 주지 않도록 CNN은 pooling을 마스킹하고 LSTM은 실제 길이로 패킹했습니다. 세 seed를 사용했고, 검증 Macro-F1로 checkpoint를 선택한 뒤 테스트했습니다. 답변 256토큰은 두 조건에서 동일하게 유지하고, 질문을 추가할 때는 별도로 최대 64토큰을 붙였습니다.”

## 4장 — 기본 결과 (40초)

“답변만 사용하는 CNN 평균 정확도는 {pct(cnn)}, BiLSTM은 {pct(lstm)}였습니다. 세 번의 학습 평균과 표본 표준편차를 함께 보고했습니다. 질문만 사용하는 조건은 두 모델 모두 33.3%여서 데이터 라벨이 질문에 새지 않았다는 통제 결과와 맞습니다. TF-IDF 기준선도 함께 비교했습니다. 높은 정확도 자체보다 공정한 조건과 재현 가능한 평가가 중요하다고 보았습니다.”

## 5장 — 질문을 추가하면 좋아지는가? (45초)

“질문과 답변을 합쳤을 때 CNN의 평균 변화는 {pp(cdelta['delta'])}, 95% 구간은 {ci(cdelta['ci95'])}입니다. {cnn_uncertainty} BiLSTM 변화는 {pp(ldelta['delta'])}, 구간은 {ci(ldelta['ci95'])}입니다. 같은 테스트 질문과 같은 학습 seed를 짝지어 비교했습니다. BiLSTM이 마지막 상태를 이용하기 때문에 질문을 앞에 붙이면 역방향 상태와 시퀀스 길이가 달라지는 것이 가능한 이유입니다. 다만 이것을 원인으로 증명한 실험은 아니므로 별도 인코더나 attention 실험이 후속 과제입니다.”

주의: 이 구간은 고정된 테스트 표본의 불확실성을 나타냅니다. seed를 재표집하지 않으며, 여러 비교에 대한 확정적 유의성 주장을 하지 않습니다. 성능을 보고 다시 조정한 결과가 아닙니다.

## 6장 — 다른 분포에서도 되는가? (40초)

“RQ3는 두 가지로 확인했습니다. 첫 번째는 출처가 확실한 SelfInstruct 질문을 학습에서 전부 빼고 평가하는 방식입니다. 두 번째는 질문 내용만으로 정한 writing 분야를 전부 빼는 방식입니다. 두 결과는 seed42를 이용한 탐색적 실험입니다. 기본 실험과 학습 데이터 수 및 테스트 분포가 달라지므로, 점수 차이가 도메인 때문이라고 단정하지 않았습니다.”

추가 설명: SelfInstruct는 검증된 원본 출처이며 writing은 휴리스틱 내용 분류입니다. 둘을 같은 의미의 ‘도메인’이라고 뭉뚱그리지 않습니다. 새로운 LLM 계열이나 현재 최신 버전을 식별했다는 의미도 아닙니다.

## 7장 — 어떤 특징을 이용하는가? (40초)

“길이, 어휘 다양성, 줄바꿈, 구두점, Markdown 같은 16개 특징만으로 학습한 모델은 {pct(feature['accuracy'])}를 얻었습니다. 각 조건에서 한 가지 변환을 적용하고 모델을 다시 학습했지만, 여러 신호가 함께 바뀔 수 있습니다. 답변을 64토큰으로 제한하는 실험과 형식을 정규화하는 실험을 따로 수행했습니다. 또한 학습 데이터만으로 만든 TF-IDF의 큰 양의 계수를 보여드렸습니다. 이 단어들은 기준선이 활용한 연관성이고 CNN이나 LSTM의 내부 판단을 직접 설명하는 것은 아닙니다.”

추가 설명: 길이 제한은 길이 차이와 정보량을 함께 바꾸며, 형식 정규화는 markup·토큰화·길이를 함께 바꿀 수 있습니다. Markdown 제거는 완벽한 문체 제거가 아닙니다. 남은 성능을 보편적인 모델 지문으로 해석하지 않습니다.

## 8장 — 한계와 배운 점 (35초)

“이번 데이터에서는 답변에 모델 계열을 구분할 신호가 있었고, 질문을 추가하는 효과는 구조에 따라 달랐습니다. 데이터 출처, 길이와 형식이 결과에 영향을 줄 수 있다는 점도 확인했습니다. 사용 모델은 과거 버전이고, 원본 생성 설정의 길이 제한도 서로 달랐습니다. 코드, 출처, 38개 실제 실행 결과, 예측과 checkpoint를 함께 제공해서 결과를 재현할 수 있게 했습니다. AI 도움은 보고서에 공개했으며, 저희가 결과와 설계를 이해하고 설명하는 것이 제출 책임입니다.”

## 예상 질문에 대한 짧은 답

- **왜 질문만은 33.3%인가요?** 같은 질문이 세 가족에 한 번씩 있으므로 결정적인 질문 전용 분류기는 각 질문에서 정확히 한 개만 맞힙니다.
- **CNN이 무조건 더 좋은가요?** 이번 데이터와 작은 모델 설정에서만 얻은 결과입니다. 모든 RNN이나 최신 LLM에 일반화하지 않습니다.
- **왜 Macro-F1도 쓰나요?** 각 모델 계열의 F1을 같은 비중으로 평균해 특정 계열에 치우친 예측을 확인합니다. 단일 라벨 문제에서 Micro-F1은 accuracy와 같습니다.
- **새로운 모델도 알 수 있나요?** 아닙니다. 세 라벨 중 하나를 고르는 closed-set 분류기이므로 unknown 판별 기능은 없습니다.
- **직접 답변을 생성했나요?** 공개 벤치마크의 공식 모델 출력을 사용했고, 원본 revision·모델 ID·출처·제외 규칙·해시를 기록했습니다.
- **AI가 도왔나요?** 데이터 처리, 구현, 실험 실행과 문서 준비에 도움을 받았으며 실제 실행 숫자만 보고했습니다. 제출자는 코드와 결과를 이해하고 설명해야 합니다.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    pdf, notes = build(args.root)
    print(f"Built {pdf} (8 slides) and {notes}")


if __name__ == "__main__":
    main()

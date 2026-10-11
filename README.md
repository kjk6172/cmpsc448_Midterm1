# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project, October 11, 2026. **Team leader: Kiseong Kim. Team member: Jisung Park.**

This project implements a text CNN and a bidirectional LSTM to identify the LLM family behind a response. It uses authentic, previously published AlpacaEval outputs from GPT-4-0613, Claude 2 and Gemini Pro. Every retained question has one answer from each family: **448 matched questions and 1,344 responses**, balanced exactly by family. Exact and rule-detected near-duplicate question templates remain together across splits.

The submission report is [report/project_report.pdf](report/project_report.pdf). The matching [Markdown report](report/project_report.md), saved predictions, configurations and execution histories make its measurements reviewable. [The Korean defense guide](docs/DEFENSE_GUIDE_KO.md) explains the project and common presentation questions. Optional presentation materials are included: [eight-slide PDF](report/presentation.pdf) and [Korean speaker notes](docs/PRESENTATION_NOTES_KO.md).

## Experiments

| Study | Conditions | Repetitions |
| --- | --- | --- |
| RQ1 / RQ2, required | CNN and BiLSTM, each with prompt only / response only / prompt + response | Seeds 42, 43, 44: 18 neural runs |
| Main references | TF-IDF logistic regression for all three input conditions; style-feature logistic regression; majority and uniform-random controls | Seed 42: 6 runs |
| RQ3, source transfer | Self-Instruct source excluded from training; TF-IDF in three modes and response-only CNN/BiLSTM | Seed 42: 5 runs |
| RQ3, task transfer | Heuristic writing category excluded from training; TF-IDF in three modes and response-only CNN/BiLSTM | Seed 42: 5 runs |
| RQ4, ablations | Response-only CNN/BiLSTM, first-64-token budget and formatting-normalized text, each retrained | Seed 42: 4 runs |

The full protocol produces **38 completed runs**. Tables report measured accuracy and Macro-F1; the main neural results include mean and sample SD across seeds. Paired uncertainty intervals resample the same `split_group_id` components for both conditions. See `artifacts/statistical_analysis.json` for the complete calculations.

Prompt-only accuracy is a deliberate negative control. Each test question occurs once under each of the three balanced labels, with identical input text, so a deterministic classifier must achieve exactly 1/3 accuracy. Macro-F1 can be lower if predictions collapse onto one family.

<!-- MEASURED_RESULTS_START -->

### Measured main results

The same 261 held-out response rows are used for all main conditions. Values are means over three seeds; sample SD is in percentage points.

| Input | Model | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Prompt only | CNN | 33.3% +/- 0.0 pp | 29.6% +/- 1.2 pp |
| Prompt only | BiLSTM | 33.3% +/- 0.0 pp | 32.2% +/- 0.6 pp |
| Response only | CNN | 73.8% +/- 2.5 pp | 73.5% +/- 2.4 pp |
| Response only | BiLSTM | 75.4% +/- 1.5 pp | 75.4% +/- 1.5 pp |
| Prompt + response | CNN | 76.4% +/- 1.5 pp | 76.0% +/- 1.7 pp |
| Prompt + response | BiLSTM | 54.2% +/- 2.5 pp | 54.1% +/- 2.4 pp |

CNN context change: +2.6 pp, paired 95% interval [-0.5, +5.7] pp; the interval includes zero, so a directional difference is not established. BiLSTM context change: -21.2 pp, paired 95% interval [-26.2, -16.0] pp; the interval excludes zero in this fixed test sample. Source and writing holdouts show the limits of distribution transfer. Full results and intervals are in the PDF and saved JSON.

<!-- MEASURED_RESULTS_END -->

## Reproduce on Windows PowerShell

The recorded environment uses **Python 3.12 and CPU PyTorch**. Use a Python 3.12 interpreter, then run these commands from the repository root. Retraining all models takes time; saved results and seed-42 checkpoints are included for review and inference.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe scripts\prepare_data.py --download
.\.venv\Scripts\python.exe scripts\run_experiments.py
.\.venv\Scripts\python.exe scripts\run_additional_domains.py
.\.venv\Scripts\python.exe scripts\analyze_results.py
.\.venv\Scripts\python.exe scripts\build_report.py
.\.venv\Scripts\python.exe scripts\build_slides.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_submission.py
```

If `py -3.12` is unavailable, use the installed Python 3.12 executable to create the environment. `requirements.txt` gives the direct project dependencies; `requirements-lock.txt` records the exact installed package versions. On macOS/Linux, create the environment with `python3.12 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in the commands above.

Data preparation downloads pinned upstream snapshots, validates SHA-256 checksums, then applies deterministic fixed filters. Subsequent runs use the ignored local raw cache. The curated CSV and JSONL are included, so the downloaded files are not needed merely to inspect the data or train on the included subset. A complete data reconstruction requires network access.

`run_experiments.py` resumes already completed neural runs only if the data hash, implementation hash, split and training configuration match. `--force` retrains them. The standard main split is fixed at seed 2026; seeds 42/43/44 vary model initialization and minibatch order. `run_additional_domains.py` appends the writing holdout without replacing the main/source-transfer results.

The 12 unit tests cover grouping, balanced prompt controls, token budgets, padding invariance and core pipeline behavior. `verify_submission.py` independently recomputes all 38 run metrics from saved predictions, checks required conditions and split isolation, replays all test predictions from 14 neural checkpoints, and inspects the PDF's author names, research questions and figure bounds. Its machine-readable outcome is `artifacts/verification.json`. `build_slides.py` produces the optional eight-slide PDF and Korean notes from the saved measurements.

## Inference from a saved model

For response-only CNN inference, put UTF-8 response text in `answer.txt` and run:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -m llm_attribution.predict --checkpoint artifacts\runs\main_cnn_output_raw_seed42\checkpoint.pt --answer-file answer.txt
```

The output includes a predicted family and three uncalibrated softmax probabilities. A combined-input checkpoint also accepts `--question "Your question"`. This is a closed-set classifier for the three historical generators, not a reliable detector of arbitrary human or AI authorship.

## Implementation and controls

- **CNN:** trainable 64-dimensional embeddings; Conv1d kernels 3/4/5 with 48 filters each; ReLU; padding-aware global max pooling; dropout 0.3; three-class linear head.
- **BiLSTM:** trainable 64-dimensional embeddings; one bidirectional LSTM with 48 hidden units per direction; packed sequences; final forward/backward states; dropout 0.3; three-class linear head.
- **Optimization:** AdamW, learning rate 0.002, weight decay 0.0001, batch size 32, gradient norm clipping at 1.0, at most 20 epochs, early-stopping patience 4. Selection uses validation Macro-F1, with validation loss as a tie-breaker.
- **Tokenization:** lowercased word-like tokens and punctuation, a training-only vocabulary of at most 20,000 tokens with minimum frequency 2, and reserved PAD/UNK/SEP tokens.
- **Fair response budget:** prompt only uses at most 64 question tokens. Response only uses at most 256 answer tokens. Combined input prepends up to 64 question tokens and a separator to the same 256-answer-token budget, for at most 321 tokens.
- **Splits:** 429 exact/near-duplicate template components; main train/validation/test have 278/65/86 components and 278/83/87 questions. All family answers follow their question/component. Source and writing holdout models are fitted anew. Koala and OASST are absent from every split.
- **Reference controls:** TF-IDF and feature classifiers use full text, not the neural token budgets. Their regularization is selected from C = 0.1/1/10 on validation Macro-F1. Their higher/lower scores are practical references, not an exactly matched parameter-budget comparison.

## Data provenance and license

The dataset comes from the [official AlpacaEval repository](https://github.com/tatsu-lab/alpaca_eval), pinned at `cd543a149df89434d8a54582c0151c0b945c3d20`. The [source manifest](data/source_manifest.json) records immutable URLs, model configurations, hashes, curation criteria and limitations. The [data documentation](data/README.md) and [notice](data/NOTICE.md) preserve source attribution.

Starting with 805 matched questions, curation excludes 188 OpenAssistant-source groups and all 156 Koala-source groups, then 1 blank-answer group and 12 conservative contact/address-pattern matches. Koala is excluded to honor the upstream request against training use. All 38 final runs were retrained on this revised subset. Model/provider identity substrings are replaced with the common `[MODEL]` token. No subjective answer quality or classification performance is used to select examples. `data/processed/audit.json` contains the exact counts.

**The processed benchmark data remains CC BY-NC 4.0 and is intended for noncommercial academic use.** The project's code license does not override the dataset license. Upstream code is Apache-2.0; data and code notices are separate.

The source holdout measures benchmark-collection transfer; the writing holdout uses fixed prompt-keyword rules, not expert task annotations. Only one historical snapshot per provider is represented, and generation caps differ (GPT/Claude 4,096 versus Gemini 2,048 output tokens). Performance does not establish universal fingerprints or transfer to current versions, unknown families, humans or edited text.

## Repository map

```text
data/processed/              Curated CSV/JSONL and deterministic curation audits
data/source_manifest.json    Source snapshots, licenses, methods and hashes
data/licenses/               Preserved upstream data/code license notices
src/llm_attribution/         Models, tokenizer, splitting, training and inference
scripts/                    Data preparation, experiments, analysis and report generation
artifacts/results.csv        One record per completed experiment
artifacts/results.json       Detailed run metrics and experiment metadata
artifacts/main_aggregate.csv Required neural seed means and sample SDs
artifacts/split.json         Exact main prompt membership and split-unit description
artifacts/runs/              Predictions, metrics, histories, configs and saved models
report/                     Final PDF, readable Markdown and actual-result charts
docs/                       Submission checklist and Korean understanding guide
metadata.json               Confirmed authors, course, date and AI disclosure
requirements-lock.txt       Recorded dependency versions
```

The raw download cache, virtual environment and local tooling are excluded from Git and the submission package.

## AI assistance and submission

Codex assisted with data curation, implementation, experiment execution and manuscript preparation. The reported measurements come from the included code and saved outputs. Both students are responsible for reviewing, understanding, verifying and explaining the submitted work. No invented division of student contributions is asserted.

The final code, curated data, experiment artifacts and report are published in [kjk6172/cmpsc448_Midterm1](https://github.com/kjk6172/cmpsc448_Midterm1). Anonymous access to the source and PDF has been verified, and a fresh GitHub clone passed the 12 unit tests and all artifact checks. Both students should review and understand the final report, then submit **https://github.com/kjk6172/cmpsc448_Midterm1** through the course's required channel. Uploading to GitHub does not itself submit the assignment. The deadline is October 11, 2026, 11:59 p.m. America/New_York. See [the submission checklist](docs/SUBMISSION_CHECKLIST.md).

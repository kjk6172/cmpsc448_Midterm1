# Data provenance and use

This project uses authentic, pre-generated answers from the official AlpacaEval repository. Each of 448 retained questions has one response from GPT-4-0613, Claude 2 and Gemini Pro: 1,344 responses, exactly 448 per family. **Koala and OASST are excluded from all partitions.**

- Source: https://github.com/tatsu-lab/alpaca_eval
- Pinned revision: `cd543a149df89434d8a54582c0151c0b945c3d20`
- Dataset card: https://huggingface.co/datasets/tatsu-lab/alpaca_eval
- Dataset-card revision: `2edc6fad8be6b14ea7230aabfd08188da6b8b814`
- Data license: CC-BY-NC-4.0, distinct from the upstream Apache-2.0 code license.

`source_manifest.json` records immutable source URLs, hashes, model identities, filtering rules and limitations. `processed/audit.json` records exact counts and processed-file hashes. `processed/excluded_prompt_audit.json` records excluded hashes and reasons without reproducing excluded text.

## Reproduce

Run `python scripts/prepare_data.py --download` from the repository root. Data preparation uses the standard library (Python 3.11+); the full project reproduction environment is Python 3.12. Source snapshots are SHA-256 checked. Later preparations use the ignored raw cache; `--force-download` retrieves the same pinned sources again. Raw caches are excluded from Git and packaging.

## Fixed curation

1. Exclude all 188 OASST prompt groups as a conservative choice for user-contributed conversations.
2. Exclude all 156 original Koala groups before any splitting or fitting. The [Koala README](https://github.com/arnav-gudibande/koala-test-set) requests no training use; the final dataset contains no Koala-derived text in training, validation or testing.
3. After source exclusions, remove 1 group with a blank answer and 12 whole matched groups with contact/address patterns (2 email, 9 phone-like, 1 street-address). These may be fictional patterns, not proven real identities.
4. Replace model/provider identity substrings with shared `[MODEL]`: 0 inputs and 5 answers (4 Claude, 1 GPT, 0 Gemini). Original hashes preserve traceability.
5. Verify identical input text and exactly one answer per family per question. No canonical duplicate prompts remain.

No response quality, model accuracy or correctness is used in selection. The 38 final experiments were retrained after the Koala exclusion; earlier candidate results are superseded. Response truncation occurs in the model pipeline, not curation.

| Retained source | Questions | Responses |
| --- | ---: | ---: |
| helpful_base | 127 | 381 |
| selfinstruct | 242 | 726 |
| vicuna | 79 | 237 |
| Total | 448 | 1,344 |

## Schema and leakage prevention

CSV columns include `prompt_id`, `llm_name`, `llm_input`, `llm_output`, `domain`, `source_dataset`, `split_group_id` and traceability fields. JSONL also includes the assignment's `LLM_name`, `LLM_Input`, `LLM_output` spellings. Labels and generator metadata never enter classifier text.

Split on `split_group_id`. Input-only rules require at least eight word tokens, length ratio >=0.60, and character similarity >=0.85 or word-trigram Jaccard >=0.80. Connected components produce 429 groups; 3 contain multiple questions, with largest size 15. This is heuristic template deduplication, not complete semantic deduplication.

Identical questions occur equally under the three labels, so deterministic prompt-only classification must have exactly 1/3 accuracy. Macro-F1 may be lower for uneven predictions.

`domain` uses fixed ordered prompt-only keyword rules: general (323), mathematics (4), programming (45), science (8), writing (68). These are approximate tags, not expert annotations. `source_dataset` denotes benchmark collection and measures a different shift from semantic task holdout.

## Interpretation and reuse limits

Only one historical snapshot per provider is represented. Provider prompts and generation settings differ; GPT/Claude output caps are 4,096 versus Gemini 2,048 tokens. Results do not establish current-model or unknown-provider detection. The included duplicate-answer audit records any exact repeated responses; template grouping does not guarantee every generic refusal is unique across splits.

Preserve `NOTICE.md`, source attribution, indications of changes and the CC-BY-NC-4.0 noncommercial-use terms when redistributing.

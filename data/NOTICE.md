# Attribution and data license

This directory contains a filtered and identity-scrubbed noncommercial academic derivative of AlpacaEval. Changes include source exclusion, conservative privacy-pattern filtering, empty-answer filtering, shared model-name masking, deterministic task tags, and grouping of near-duplicate prompts. The downloaded source answers are authentic published benchmark outputs; this project does not assert that their substantive content is correct.

**AlpacaEval attribution:** Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. *AlpacaEval: An Automatic Evaluator of Instruction-following Models.* 2023. https://github.com/tatsu-lab/alpaca_eval

**Constituent benchmark sources retained:**

- Yizhong Wang and collaborators. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* https://github.com/yizhongw/self-instruct
- LMSYS. *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality.* https://lmsys.org/blog/2023-03-30-vicuna/
- Anthropic. *HH-RLHF helpful-base evaluation data.* https://huggingface.co/datasets/Anthropic/hh-rlhf

The upstream data license is **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)**. The legal text is included in [licenses/CC-BY-NC-4.0.txt](licenses/CC-BY-NC-4.0.txt), and the official license is https://creativecommons.org/licenses/by-nc/4.0/ . Attribution, indication of changes, and the noncommercial restriction must be retained.

The upstream repository's **code** uses Apache License 2.0, a copy of which is included in [licenses/ALPACA_EVAL_APACHE-2.0.txt](licenses/ALPACA_EVAL_APACHE-2.0.txt). The processed data is not relicensed under the code license.

## Koala exclusion

The [Koala evaluation-set README](https://github.com/arnav-gudibande/koala-test-set) states: "Please do not use this test set for training purposes". All 156 Koala-source prompt groups in the original matched snapshots are excluded before splitting. None of their text is present in the final training, validation or test dataset. All 38 final experiments were retrained on the revised subset. The excluded source is cited to explain curation, not as a retained training source. The ignored local raw cache and historical candidate backup are not distributed.

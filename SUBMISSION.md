# FTEC5660: ACE Formula Reproduction

This submission reproduces an ACE context-adaptation behavior on the Formula financial reasoning benchmark and evaluates one isolated execution optimization: omitting the post-curation answer used only for training diagnostics. It uses the authors' implementation, rather than reimplementing ACE from the paper. No model weights are trained.

The original implementation is [ace-agent/ace](https://github.com/ace-agent/ace), pinned at `82709de050e1db6e6ef2f07bcb0393560b94992a`. The corresponding paper is [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618v3). Upstream method, prompts, preprocessing, scorer and `uv.lock` are preserved. Reproduction additions are under `repro/`; this document and `report/` contain the submission material. The original [Apache-2.0 license](LICENSE.txt) and upstream attribution are retained. The original repository README remains available in [README.md](README.md).

## Submission contents

| Item | Location | Purpose |
|---|---|---|
| Written report | [report/main.pdf](report/main.pdf) | Academic report, including references, within the 10-page limit |
| Complete LaTeX source | [report/main.tex](report/main.tex), [report/references.bib](report/references.bib), [report/figures/](report/figures/) | Editable report and self-contained figures |
| Reproduction implementation | [repro/](repro/) | Provider adapter, controlled variants, protocols, analysis and tests |
| Frozen protocols | [repro/configs/protocol.json](repro/configs/protocol.json), [V3 contract](repro/artifacts/v3_contract.json), [V4 contract](repro/artifacts/v4_contract.json) | Parameters, cohorts, source hashes and decision rules |
| Results and evidence | [repro/artifacts/](repro/artifacts/) | Public predictions, selected playbooks, accounting, audits and plots |
| Detailed experiment notes | [repro/README.md](repro/README.md) | Chronological procedures, recovery behavior and limitations |

The Zoom presentation is outside this delivery scope and is not included.

## Install the original locked environment

Use a full Git clone of this repository. Do not use a shallow clone: the upstream ancestry and source-byte checks require the pinned original commit to be present. The tested experiment interpreter was Python 3.12.13. Install `uv` if it is not already available, then run:

```bash
git clone https://github.com/HallwayinCHN/ace-formula-reproduction.git
cd ace-formula-reproduction
uv sync --python 3.12 --locked
```

This uses the original `uv.lock`; it does not upgrade or regenerate experimental dependencies. Formula does not require initializing the AppWorld submodule. The first installation needs network access to download packages. Some tokenization or execution paths also download the public `cl100k_base` vocabulary on first use. The project sets a local vocabulary cache at `../.runtime-cache/tiktoken`. No embedding-model download is needed for the TF-IDF variant.

## Verify the completed results without an API key

After dependency installation and the initial public tokenizer-vocabulary download, the following checks run offline using committed artifacts and the original scorer. They do not contact DeepSeek or require the ignored private request logs:

```bash
.venv/bin/python -m repro.v2_recompute
.venv/bin/python -m repro.v3_recompute
.venv/bin/python -m repro.v4_recompute
```

| Check | Evidence verified | Expected result |
|---|---|---|
| `v2_recompute` | 600 predictions on the second official 100-question test cohort, original labels, accuracy, accounting and paired interval | `status: public_rows_recompute_passed`, `new_api_calls: 0` |
| `v3_recompute` | 1,989 usage records, six training summaries/traces/playbooks, validation selection and 240 development predictions | `verified: true`, `api_calls: 0` |
| `v4_recompute` | 960 predictions on 160 project holdout questions, labels, frozen hashes, usage, costs and the primary accuracy decision | `verified: true`, `api_calls: 0` |

`v2_recompute` also regenerates the small deterministic summary `repro/artifacts/v2_public_recompute.json`. V3 and V4 verification print their conclusions. All three checks were executed successfully during submission preparation.

For the first official test cohort, the complete public rows are in [stage3_scored.jsonl](repro/artifacts/stage3_scored.jsonl), with [analysis](repro/artifacts/stage3_analysis.json) and the original [audit](repro/artifacts/stage3_audit.json). The historical `verify_stage3` command additionally requires private local request records; it is not a fresh-clone public verification command.

Offline implementation tests can be run separately:

```bash
.venv/bin/python -m unittest discover -s repro/tests -v
```

## Read the main results

| Question | Matched comparison | Finding | Evidence |
|---|---|---|---|
| Does accumulated ACE context help in this setup? | Empty-playbook Generator versus ACE on the first official 100-question test cohort; three training-order seeds | 69.33% versus 84.00%; difference +14.67 percentage points, question-clustered 95% interval [+8.33, +21.67] | [STAGE3.md](repro/artifacts/STAGE3.md) |
| Does removing the diagnostic preserve the learning path under fixed responses? | Original versus diagnostic-free replay of archived retained responses | All 180 step states, six validation checkpoints and three selected playbooks match; no new API calls | [DIAGNOSTIC_FREE_RESULTS.md](repro/artifacts/DIAGNOSTIC_FREE_RESULTS.md) |
| What happens in fresh training runs? | Three paired original/diagnostic-free runs, 60 training and 30 selection-validation questions each | Learning calls 787 to 602 (-23.51%); cached learning-cost estimate 5.9796 to 4.9114 CNY (-17.87%); median workflow time 788.6 to 646.2 seconds (-18.06%) | [V3_RESULTS.md](repro/artifacts/V3_RESULTS.md) |
| Is accuracy preserved on the project holdout? | Six V3-selected playbooks, original versus diagnostic-free training, 160 holdout questions | Each group 415/480 (86.46%); difference 0.00 percentage points, question-clustered 95% interval [-2.08, +2.08]; the predefined interval criterion was not met | [V4_RESULTS.md](repro/artifacts/V4_RESULTS.md) |
| Did the additional proposals improve the trade-off? | Selective reflection, retrieval, brief output and conservative routing, within their own cohorts | Mixed or negative quality results; routing failed its development cost gate and was not evaluated on holdout | [STAGE3.md](repro/artifacts/STAGE3.md), [V2_RESULTS.md](repro/artifacts/V2_RESULTS.md), [V3_RESULTS.md](repro/artifacts/V3_RESULTS.md) |

The main report focuses on ACE behavior reproduction and diagnostic-free execution. The first study's prespecified primary modification comparison was selective reflection plus retrieval versus ACE. Diagnostic-free training became the report's main engineering contribution after the earlier studies; V3 and V4 have their own prospectively frozen protocols. This chronology is preserved rather than rewritten as one original hypothesis.

## Optional: configure and run a new live pilot

No credentials are needed to inspect the submitted evidence or perform the public recomputations. To run new model requests, create a local ignored configuration file:

```bash
cp repro/deepseek.env.example .env
chmod 600 .env
```

Open `.env` locally and fill in the value of `DEEPSEEK_API_KEY`. Do not commit the file or paste the key into command arguments, issue text or logs. Check readiness without making model requests:

```bash
.venv/bin/python -m repro.pilot --check
```

The live pilot below makes **paid DeepSeek API calls** and creates a new run on every invocation. It uses 10 training and 10 validation questions, separate from formal study cohorts, with a 20-CNY accounting stop line and the project-wide 180-CNY accounting stop line. These are conservative ledger controls, not guarantees about provider billing:

```bash
.venv/bin/python -m repro.pilot --live
```

The provider configuration is `deepseek-flash` at `https://api.deepseek.com`, temperature 0, thinking disabled and a 4,096-token output cap. The exact model parameters, pricing assumptions and the permissive handling of nonempty truncated responses are recorded in [repro/configs/pilot.json](repro/configs/pilot.json). Formal experiments use 60 training questions, 30 selection-validation questions and order seeds 42/43/44, with validation checkpoints after steps 30 and 60. See [repro/README.md](repro/README.md) for the staged execution commands. Do not overwrite the frozen protocols or treat previously evaluated questions as a new unseen test.

New live runs may differ because provider responses, model aliases, cache state and network conditions are not fully controlled. A fresh clone does not contain historical private response caches, and cannot resume the original paid jobs byte-for-byte. Public artifact verification is the appropriate starting point for reviewing the submitted results.

## Build the report

The LaTeX project is contained in `report/`. With a LaTeX distribution and `latexmk` installed:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The result is `report/main.pdf`. References and figures are stored with the source; the scientific figures do not need an API key or a model call. The report can also be uploaded as a complete folder to a LaTeX editor.

As an alternative to a full TeX distribution, a Tectonic installation can build the same source with `tectonic main.tex` from `report/`; its first build may need network access to obtain TeX packages.

## Interpretation and reproducibility boundaries

- **Behavior reproduction, not an exact paper-number replication.** The paper's Formula Table 2 uses DeepSeek-V3.1 and a different experimental scale. This project uses DeepSeek Flash and a reduced subset; its empty-playbook baseline uses the upstream ACE Generator prompt. Paper and project numbers are therefore not directly interchangeable.
- **Cohorts remain separate.** The first and second official test cohorts each contain 100 different questions. V4 uses 160 questions from a separately assigned validation-pool project holdout; it is not a new official test set. Do not compare 84.00%, 80.67% and 86.46% as a sequence of gains.
- **Equal means are not proof of non-inferiority.** V4 has 160 independent question units, with three paired fixed-playbook responses per question. The observed equality did not satisfy the preregistered interval lower-bound criterion. The interval does not cover all uncertainty from future retraining.
- **Replay equivalence is conditional.** The local `repro.diagnostic_free_replay` check needs ignored historical response archives. It confirms identical states only when retained responses are held fixed; it does not guarantee identical outputs in new online calls. The committed replay result is available for inspection.
- **The optimization is measurement-overhead removal.** It does not introduce a new ACE learning algorithm. The diagnostic can be removed in this offline driver because its answer does not update learning state; this does not justify deleting an online answer that is part of the actual task output. Skipped training diagnostic accuracy is marked as unmeasured, not as zero or copied from the initial answer.
- **Data-boundary deviation is disclosed.** V3's legacy loader parsed the full validation file before selecting the original 30 records. The remaining records were not used in actual training/validation model prompts. V4 selects physical row IDs before parsing and audits all 960 actual prompts. The holdout must not be described as having never been parsed in earlier processes.
- **Scoring and failures are retained.** Original Formula labels and exact numeric matching are unchanged. Nonempty truncated answers remain in the fixed evaluation denominator. Ambiguous units or possible label issues are discussed rather than corrected after seeing results.
- **Costs are estimates, not reconciled bills.** Cache-aware accounting uses the recorded provider usage, frozen peak prices and an 8-CNY-per-USD budgeting factor. Standardized uncached cost is a separate metric. Unknown-billing reserves remain in the project ledger. Training and inference saving percentages must not be added, and cross-stage assembled costs are not a single continuous deployment bill.
- **Fresh-run differences have several causes.** Retained learning calls were 607 versus 602 in V3, and one seed's learning-cost estimate increased. New responses, reflection trajectories and cache effects prevent attributing every net difference solely to the deleted diagnostic.

## Repository hygiene

Private `.env` files, `.venv/`, `.plot-venv/`, local `results/`, request recovery caches and operating-system files are excluded by `.gitignore`. Committed usage exports are restricted to experiment metadata and do not include authorization headers or API keys. Source history and staged commits are retained so that experimental additions can be reviewed against the original pinned implementation.

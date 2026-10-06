# CTREvo

[中文](README.zh-CN.md) · [Design](docs/design.md) · [Dataset selection](docs/datasets.md)

**An Agent that edits CTR model code, runs controlled experiments, and learns from the results.**

CTREvo connects [ModelEvoHarness](https://github.com/CharlesXu-HQ/ModelEvoHarness) to a concrete click-through-rate research task. The Agent can add interaction branches inside a backbone, change their fusion, revise a loss, or reject a hypothesis after evaluation. Every proposed change includes its rationale, comparison, and falsification criterion.

The goal is to improve the yield of useful experiments. Metric gains must be measured; they are not guaranteed by using an Agent.

## Measured first run

Full Criteo, RTX 5090, one epoch per candidate: the selected MLP + Cross recipe improved independent test **LogLoss 0.451195 → 0.447231** and **AUC 0.802677 → 0.806719**. The first FM candidate worsened metrics; the next local edit improved them. The run required format fixes and recovery, with 24 provider calls across setup and search. [Full evidence, intervals and limitations](docs/experiments.md).

## What the Agent can change

| Research surface | Implementation |
|---|---|
| Representation and interactions | Native PyTorch `build_model(schema, config)`; editable source, embeddings, interaction layers and parallel subnetworks |
| Composition | Harness source references, explicit component identities, branch inputs, parameter sharing and fusion |
| Training objective | Optional `training_loss(logits, labels)`; learning rate and weight decay within fixed bounds |
| Research decisions | Evidence-backed priorities, alternatives, source reads, local iteration and selective inheritance |
| Experience | Dataset-bound technical findings; business conclusions only when the fields actually support them |

The host fixes the dataset, splits, training-row coverage, optimizer family, batch size, epoch ceiling and evaluator. It records actual CUDA execution and metrics separately from the Agent's claims. A successful run does not prove that each component caused a gain.

## One concrete benchmark

The initial adapter uses **all 45,840,617 labeled rows** of Criteo's Display Advertising Challenge release: **13 numerical + 26 categorical original features**. Thirteen missing indicators are derived inputs, not additional original fields.

- Fixed original-order 80/10/10 train/validation/test split; timestamps are unavailable, so this is **not a verified chronological split**.
- Signed `log1p` numerical transformation; normalization fitted on training only. Deterministic field-specific categorical hashing, 65,536 buckets per field, missing reserved as zero.
- Primary objective: validation **LogLoss**. Secondary: AUC, Brier score and calibration.
- Candidate comparisons use paired per-impression LogLoss differences. Their normal intervals do not correct for repeated adaptive selection or unknown user clustering.
- Final test evaluation is separate, using frozen selected and baseline checkpoints. The Agent never receives test metrics during search.
- Anonymous fields cannot support user-interest sequence claims or interpretable business segments. DIN is not assumed applicable simply because categorical fields exist.

## Experiment loop

```text
fixed task + prior observations + Harness guidance
    → inspect relevant source → hypothesis + executable candidate
    → isolated full-data CUDA training → independent validation metrics
    → compare with hypothesis → component-level experience → next decision
```

ModelEvoHarness is a Git submodule at `third_party/model-evo-harness`. Its recorded upstream commit makes runs reproducible. Updating that dependency is an explicit project maintenance action.

### Implementation checks before promotion

CTREvo defaults to strict promotion: a better score becomes the selected candidate only after the host verifies **bounded component execution**. The first three real training batches observe declared modules, container children, bound methods and custom `candidate.training_loss` functions. The host records output gradients and module parameter gradients, checks that the model parameter set still matches optimizer construction, and independently enforces full-data CUDA coverage. Missing paths are contradicted; unobserved paths remain unverified and available for diagnosis.

This does not prove a named architecture's mathematics, declared parameter sharing, exact fusion topology, or an individual component's benefit. `change_audit` and benefit attribution remain unverified until appropriate comparisons are run. `--no-require-verified-implementation` enables exploratory score selection; use the same setting for search, resume and finalize. Policy/code changes require a new run.

[Full-data GPU replay and verification coverage](docs/implementation-verification.md): the saved FM candidate remains rejected, the saved Cross candidate passes strict promotion, and the original validation scores are reproduced.

### Evidence-led explicit crosses

The Agent now audits numeric, categorical and missingness views before choosing an interaction. It can extend the current backbone with selected field-group terms, or defer structure work for diagnosis. No field type or FM branch is automatically first. Bounded output-scale evidence supports fusion diagnosis; dataset-bound history retains which scope was actually tested.

An interaction trial executes **one additional same-source control** and records paired validation metrics. This doubles the training arms for that trial; it is part of the frozen budget. The optional [mixed-field example](examples/mixed_fm.py) uses native Harness primitives and actual configuration gates. The host keeps completed main results if a control fails. See [design, control boundaries and full-data evidence](docs/explicit-interactions.md).

## Run

Linux with NVIDIA GPU, NVIDIA Container Toolkit, a CUDA-compatible Docker image, and a Python environment compatible with that image is required. CPU-only machines can run contract tests; model experiments refuse CPU fallback.

```bash
git clone --recurse-submodules https://github.com/CharlesXu-HQ/CTREvo.git
cd CTREvo
docker build -f Dockerfile.gpu -t ctrevo-cuda .
python3.12 -m venv .venv
.venv/bin/pip install -e . -e third_party/model-evo-harness
python scripts/download.py /data/criteo
.venv/bin/ctrevo prepare --raw /data/criteo/train.txt --output /data/criteo/prepared

# Supply credentials through your environment; never commit them.
export CTR_AGENT_API_KEY='...'
.venv/bin/ctrevo search --data /data/criteo/prepared --output runs/criteo \
  --image ctrevo-cuda --venv "$PWD/.venv" \
  --provider-url https://api.deepseek.com --model deepseek-flash --steps 2

# Only after search finishes; identical protocol arguments are required.
.venv/bin/ctrevo finalize --data /data/criteo/prepared --output runs/criteo \
  --image ctrevo-cuda --venv "$PWD/.venv"
```

The Docker image must resolve the mounted venv's Python executable and native dependencies. The tested host environment, when available, is recorded with experiment results. Model availability and supported reasoning settings depend on your provider. Requests use enabled thinking, `high` for normal decisions and `max` for flagged review; `--thinking omit` supports providers without that extension.

`--resume` continues an interrupted search only when its task, protocol and Harness identity match. The default budget is one full training epoch per candidate and two attempted Agent trials, plus the seed baseline and up to one extra full-data control per interaction trial. A trial timeout is recorded as failure, never scored as partial training.

If a provider exhausts its format-repair attempts, preserve the terminal log and resume with its rejected response:

```bash
python scripts/resume_with_feedback.py --data /data/criteo/prepared --output runs/criteo \
  --error-log runs/search-terminal.log --image ctrevo-cuda --venv "$PWD/.venv" --steps 2
```

Use the same seed, epoch ceiling, batch size and timeout as the original run. The helper forwards the actual error and previous response to the Agent; it never edits candidate source or bypasses validation. Capture the original terminal output with your shell's redirection. The default `search --resume` remains available for ordinary interruptions.

## Artifacts and trust boundary

`journal.json` records proposals, host-observed source-read/delivery events, metrics, reflections, promotion decisions and the selected candidate. Agent-supplied reference hashes cannot become official read records. Delivery means source was supplied to the completion callback, not proof of model understanding or correct reuse. Each trial retains source, checkpoint, predictions, execution logs, bounded implementation checks and GPU metadata. `final/report.json` is written by the separate final evaluation. Provider logs contain final responses and usage, not API credentials or reasoning text.

Candidate containers have no network, provider key or held-out labels. They receive read-only train arrays and target **features**; scoring happens on the host. Docker isolation and source import checks reduce accidental leakage, but this is not a hardened execution service for adversarial code. The scoped implementation check and the separate **unverified** benefit attribution are both exposed to the Agent. Strict finalization refuses an unverified selection before running holdout evaluation.

Unit-test fixtures check contracts only. They are never benchmark results. See [experiment status](docs/experiments.md) for measured runs and remaining work.

## Development

```bash
python -m unittest discover -s tests
```

[Apache-2.0](LICENSE) covers this project's code. Obtain datasets from their distributors and observe their terms; raw datasets, credentials and model checkpoints are not published here.

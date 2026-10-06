# CTREvo

[中文](README.zh-CN.md) · [Design](docs/design.md) · [Dataset selection](docs/datasets.md)

**An Agent that edits CTR model code, runs controlled experiments, and learns from the results.**

CTREvo connects [ModelEvoHarness](https://github.com/CharlesXu-HQ/ModelEvoHarness) to a concrete click-through-rate research task. The Agent can add interaction branches inside a backbone, change their fusion, revise a loss, or reject a hypothesis after evaluation. Every proposed change includes its rationale, comparison, and falsification criterion.

The goal is to improve the yield of useful experiments. Metric gains must be measured; they are not guaranteed by using an Agent.

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

ModelEvoHarness is an **unmodified Git submodule** at `third_party/model-evo-harness`. Its recorded commit makes runs reproducible. Updating that dependency is an explicit project maintenance action.

## Run

Linux with NVIDIA GPU, NVIDIA Container Toolkit, a CUDA-compatible Docker image, and a Python environment compatible with that image is required. CPU-only machines can run contract tests; model experiments refuse CPU fallback.

```bash
git clone --recurse-submodules https://github.com/CharlesXu-HQ/CTREvo.git
cd CTREvo
python -m venv .venv
.venv/bin/pip install -e . -e third_party/model-evo-harness
python scripts/download.py /data/criteo
.venv/bin/ctrevo prepare --raw /data/criteo/train.txt --output /data/criteo/prepared

# Supply credentials through your environment; never commit them.
export CTR_AGENT_API_KEY='...'
.venv/bin/ctrevo search --data /data/criteo/prepared --output runs/criteo \
  --image YOUR_CUDA_IMAGE --venv "$PWD/.venv" \
  --provider-url https://api.deepseek.com --model deepseek-flash --steps 2

# Only after search finishes; identical protocol arguments are required.
.venv/bin/ctrevo finalize --data /data/criteo/prepared --output runs/criteo \
  --image YOUR_CUDA_IMAGE --venv "$PWD/.venv"
```

The Docker image must resolve the mounted venv's Python executable and native dependencies. The tested host environment, when available, is recorded with experiment results. Model availability and supported reasoning settings depend on your provider. Requests use enabled thinking, `high` for normal decisions and `max` for flagged review; `--thinking omit` supports providers without that extension.

`--resume` continues an interrupted search only when its task, protocol and Harness identity match. The default budget is one full training epoch per candidate and two attempted Agent trials, plus the seed baseline. A trial timeout is recorded as failure, never scored as partial training.

## Artifacts and trust boundary

`journal.json` records proposals, reference hashes, metrics, reflections and the selected candidate. Each trial retains source, checkpoint, predictions, execution logs and GPU metadata. `final/report.json` is written by the separate final evaluation. Provider logs contain final responses and usage, not API credentials or reasoning text.

Candidate containers have no network, provider key or held-out labels. They receive read-only train arrays and target **features**; scoring happens on the host. Docker isolation and source import checks reduce accidental leakage, but this is not a hardened execution service for adversarial code. Output checks and gradient diagnostics do not independently verify mechanism attribution; the current adapter marks it **unverified**.

Unit-test fixtures check contracts only. They are never benchmark results. See [experiment status](docs/experiments.md) for measured runs and remaining work.

## Development

```bash
python -m unittest discover -s tests
```

[Apache-2.0](LICENSE) covers this project's code. Obtain datasets from their distributors and observe their terms; raw datasets, credentials and model checkpoints are not published here.

# Full Criteo GPU experiment — 2026-10-06

## Result

The Agent completed two evaluated candidate experiments and their reflections, followed by a separate final holdout evaluation. **The selected recipe improved held-out LogLoss by 0.88% relative to the seed baseline.** Completion required adapter fixes and interrupted-run recovery; this was not an uninterrupted autonomous success.

| Evaluated recipe | Validation LogLoss ↓ | Validation AUC ↑ | Parameters | CUDA training + prediction seconds |
|---|---:|---:|---:|---:|
| Seed: field embeddings + MLP | 0.445336354 | 0.804936496 | 27,328,001 | 17.23 |
| Trial 1: shared embeddings, MLP + unweighted FM scalar | 0.448403641 | 0.801853466 | 27,328,001 | 17.39 |
| Trial 2: retained MLP trunk + two vector Cross layers + joint head | **0.441451155** | **0.808926176** | 27,720,055 | 20.36 |

These timings are worker-reported training plus prediction time, excluding container startup, host metric computation, preprocessing and Agent calls.

| Frozen checkpoint | Final test LogLoss ↓ | Final test AUC ↑ | Final test Brier ↓ |
|---|---:|---:|---:|
| Baseline | 0.451194970 | 0.802677036 | 0.146189363 |
| Selected trial 2 | **0.447230661** | **0.806719368** | **0.144836432** |

The paired final-test difference, candidate minus baseline, is **−0.003964309**, with a per-impression normal 95% interval **[−0.004056447, −0.003872171]**. Final test labels were not supplied to the Agent; selection was completed before this evaluation. The result is conditional on these fitted models and this split. Unknown user clustering and seed-to-seed variance are not represented in the interval.

## Data and protocol

- Complete Criteo Display Advertising Challenge labeled release: **45,840,617 rows**, 39 original fields; no sampling.
- Positional partitions: **36,672,493 train / 4,584,062 validation / 4,584,062 test**. Original order is not claimed to be chronological.
- The complete archive was retrieved through the [Hugging Face mirror](https://huggingface.co/datasets/Recommenders/criteo/blob/main/dac.tar.gz), transported via `hf-mirror.com`, then verified against the published 4,576,820,670-byte size and MD5 `df9b1b3766d9ff91d5ca3eb3d23bed27`. The official Azure endpoint was reachable but slower.
- Raw SHA-256, complete split-array hashes and training-only preprocessing statistics are in the [manifest](../experiments/criteo-full-20261006/dataset-manifest.json).
- PyTorch on NVIDIA GeForce RTX 5090, seed 42, AdamW, learning rate 0.002, weight decay 1e-6, batch size 8,192, one full epoch per evaluated candidate. Each consumed **36,672,493 training rows in 4,477 updates**.
- ModelEvoHarness: **unchanged** submodule commit `23c947ddc90b0a6ebafddfebad228021f1a54d5e`.
- Agent: `deepseek-flash`, enabled thinking, `high` on all completed calls in this experiment. The `max` anomaly-review route exists but was not exercised here.
- Evaluator/candidate adapter source matches commit `d0ff00a`; the later recovery helper is introduced by `5ee85fb`. The helper adds rejected-response feedback without changing the frozen evaluator or candidate code. Exact task/catalog/implementation identities are retained in the journal.

## What actually happened inside the backbone

Trial 1 retained the embedding MLP and introduced a parameterless FM scalar on its shared categorical embeddings. Its validation LogLoss became worse by 0.003067287; the paired interval was [0.002949498, 0.003185077]. The Agent recorded the negative result and uncertainty about scale and shared gradients.

Several unexecuted drafts considered a first-order linear branch or a scaled FM gate. **The accepted trial 2 used a different proposal:** it retained the embedding table, adapted the MLP to return its hidden representation, dropped the FM scalar, and added two vector cross layers:

```text
x0 = concatenate(dense, flattened field embeddings)
x_next = x + x0 * Linear(x)             # two layers
logit = Linear(concatenate(cross_output, deep_hidden))
```

The journal records this as a **local edit of trial_001**, with explicit retain/adapt/drop decisions for every parent component. Source inspection confirms both branches enter the joint prediction head. Recorded first-batch gradients are nonzero for both Cross layers and the joint head. These observations support actual execution, not isolated causal attribution.

The cross structure and fusion changed together, and removing the original scalar also changed the recipe. The improvement belongs to the **whole candidate**. The Agent's final reflection correctly leaves component attribution unverified and proposes later ablations.

## Harness behavior and remaining gaps

| Design intent | Observed evidence |
|---|---|
| Edit executable structures rather than only tune MLP widths | FM branch, then native vector Cross branch and new fusion head |
| Learn from the previous experiment | Trial 2 cites the first trial's degradation and calibration, retains/adapts/drops its components |
| Assess horizontal composition | Explicit branch groups, shared embedding input, fusion and component contracts in both designs |
| Read local reference source | Host-recorded hashes for `models/pytorch/composition.py` and `training.py` in both accepted proposals |
| Keep technical and business experience separate | Component-level technical reflections; business experience marked not observable for anonymous fields |
| Preserve independent evaluation | Held-out labels absent from candidate containers; final test evaluated only after selection |

FM and Cross code were authored by the Agent. It did **not** directly import the bundled FM/DCN implementations; the recorded source reads cover generic composition and training code. This run therefore demonstrates native composition guided by the Harness, not complete coverage of its model-reference retrieval behavior.

Exploration efficiency remains the largest weakness. Two initial searches stopped before an accepted candidate because of response syntax and schema errors. The completed search also required recovery. Across all three searches there were **24 provider calls** (4 + 4 + 16), versus two novel evaluated candidates. There were three baseline executions during setup, all with the same measured validation LogLoss. No failed draft was counted as a trained model.

Observed issues included terminal JSON delimiters, misplaced candidate/branch metadata, a reasoning-only response exhausting 32,768 output tokens, and non-verbatim metadata under `retain`. CTREvo added bounded framing repair, explicit contracts, accumulated errors and a recovery helper that preserves the rejected response. Harness validation was not loosened. Model source was not manually rewritten to obtain the result.

Not every reflection sentence is reliable: the first reflection speculated about underfitting from one last-minibatch loss versus aggregate validation loss. That comparison is insufficient evidence and is not used to support the result. Business conclusions and general claims that the Harness improves research success rates are outside what this single-seed run establishes.

## Inspect and reproduce

- [Full journal: proposals, source reads, measured evaluations, reflections](../experiments/criteo-full-20261006/journal.json)
- [Executed baseline source](../experiments/criteo-full-20261006/baseline/input/candidate.py)
- [Executed FM candidate](../experiments/criteo-full-20261006/trial_001/input/candidate.py)
- [Executed Cross candidate](../experiments/criteo-full-20261006/trial_002/input/candidate.py)
- [Independent final report](../experiments/criteo-full-20261006/final/report.json)
- [Provider usage and retry summaries; no credentials or reasoning text](../experiments/criteo-full-20261006/provider-summary.json)
- [Resume feedback provenance](../experiments/criteo-full-20261006/resume-feedback.json)

Use the README commands and preserve the protocol arguments above. Models, raw data and prediction arrays remain outside Git. Fresh Agent calls may choose different candidates; the saved candidate source and configuration in the journal define the measured recipes.

Contract tests: all **15 passed on the GPU host**. Local execution passed 14 with the GPU-only test explicitly skipped.

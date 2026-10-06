# Implementation verification — 2026-10-06

This check replays the previously published seed, FM and Cross candidates on the full Criteo partitions. It validates the new host execution checks, strict Harness promotion and source provenance. It contains **no fresh DeepSeek decisions, new optimization gain, ablation, or new holdout evaluation**.

## Measured replay

RTX 5090, native PyTorch, seed 42, batch size 8,192, one complete epoch per candidate. All 45,840,617 labeled rows remain in the frozen dataset: 36,672,493 training, 4,584,062 validation and 4,584,062 reserved test rows. The replay trains on the entire training partition and evaluates the entire validation partition, without sampling or opening test labels.

| Candidate | Validation LogLoss | Execution check | Strict promotion |
|---|---:|---|---|
| Seed embedding MLP | 0.445336354 | verified | Initial comparator |
| MLP + FM scalar | 0.448403641 | verified | Rejected: no score gain |
| MLP + Cross + joint head | 0.441451155 | verified | Promoted |

The three validation scores match the original saved run. Each candidate completes 36,672,493 training rows and 4,477 optimizer steps. Component-benefit attribution remains `unverified` for both trials.

The replay deliberately supplies the saved proposals through a deterministic callback. Its new `reference_events` record the training module actually read and supplied to that callback. Old proposal hashes do not become official records. These events do not claim that a model generated a new proposal or reused reference code correctly.

## What `verified` covers

The `declared_component_execution_v1` check establishes the following within its stated observation window:

- Candidate source matches the source observed by the worker; CUDA row coverage matches the frozen protocol.
- Each declared `instance_path` resolves to a module, container child, bound method or custom `candidate.training_loss` callable.
- During the first three real training batches, the declared outputs receive finite nonzero loss gradients, and declared module parameters receive finite gradients.
- The final registered model parameter set matches the set used at optimizer construction; changes to that set are flagged.

Paths that are absent are contradicted. Paths without sufficient observations remain unverified. The check does **not** establish a named architecture's mathematical equivalence, declared parameter-sharing relationships, exact fusion topology, all conditional paths, or causal attribution of a metric gain. No extra model forward, backward or optimization step is introduced by the probe.

CTREvo enables strict promotion by default; the generic Harness leaves it optional. Unverified candidates retain their metrics and can inform further diagnosis. They cannot become the selected candidate in strict mode, and strict finalization rejects an unverified selection before holdout evaluation. `change_audit` remains independent of execution status.

## Regression checks

- **CTREvo: 30/30 tests passed** on the GPU host, with no skips. GPU cases cover connected/disconnected components, parameterless methods, ModuleList children, custom loss, missing paths, late parameters, and cached loss callables after probe closure. A paired unit test checks identical outputs, gradients, updated parameters and CUDA RNG with/without the probe.
- **Harness complete suite: 170 tests attempted; 149 passed, 11 skipped, 10 blocked by missing TensorFlow.** The 10 errors were `ModuleNotFoundError: tensorflow` in existing TensorFlow model/loss tests; this is not a complete dual-framework test pass. The changed engine, reference, provider/CLI and composition-integration suites passed all 63 tests separately.
- Harness regressions cover default versus strict promotion, contradicted/missing/null checks, policy changes on resume, forged source records, real reads versus callback delivery, changed source content, retries and proposal-scope isolation.

## Evidence and reproduction

- [Compact full-data verification report](../experiments/implementation-audit-20261006/verification-report.json)
- [CTREvo test output](../experiments/implementation-audit-20261006/ctrevo-tests.txt)
- [Seed runtime](../experiments/implementation-audit-20261006/baseline-runtime.json), [FM runtime](../experiments/implementation-audit-20261006/trial_001-runtime.json), [Cross runtime](../experiments/implementation-audit-20261006/trial_002-runtime.json)
- [Replay script](../experiments/implementation-audit-20261006/replay.py) and [original candidate journal](../experiments/criteo-full-20261006/journal.json)

After preparing the complete release as described in the README:

```bash
python experiments/implementation-audit-20261006/replay.py \
  --data /data/criteo/prepared --image ctrevo-cuda --venv "$PWD/.venv" \
  --output runs/implementation-verification
```

Use a new output directory; existing trial evidence is not overwritten. The report records task, data, evaluator, Harness implementation and promotion-policy identities. Original benchmark artifacts remain unchanged.

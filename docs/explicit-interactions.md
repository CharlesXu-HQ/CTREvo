# Explicit feature crosses

An embedding represents a field; it is not a competing feature source. CTREvo exposes three source-bound views to ModelEvoHarness: the 13 standardized numeric inputs, 26 categorical IDs, and 13 numeric-missing indicators. The indicators reference their original numeric columns. No user, item, price or sequence semantics are inferred from anonymous field names.

## Agent decision

Every proposal assesses coarse view-pair coverage, records the code/evidence basis, and either tests a ranked interaction hypothesis or defers it for higher-value work. Candidate mechanisms remain open-ended. A field subset is expressed in the actual component's `input_fields`; coverage does not require testing or materializing every original-field pair.

Priority is evidence-based: current metric failures, branch/fusion scale, gradient observations, support, missingness, cost and task/dataset-bound history. Numeric fields are not always higher priority than categorical fields. A failed categorical FM recipe does not eliminate mixed interactions; an untested pair does not prove that it is valuable. An existing Cross network may already cover mixed inputs explicitly. Loss and optimization experiments remain available through `decision: defer`.

First-three-batch output summaries (mean, population standard deviation, maximum absolute value and finite status) receive host evidence IDs such as `trial_001.output_scales`. These bounded diagnostics are not field importance or dataset-wide distribution estimates. Probe tests check that observations preserve predictions, gradients, parameter updates and RNG state.

## Executable control

An interaction test specifies a nonempty `control.config_patch.model`. The host shallowly replaces those model configuration values in a copy of the candidate, then trains one additional arm with **identical source, seed, splits, optimizer, batch size and epochs**. Training settings cannot be patched. Identical host configuration dictionaries are rejected before training; the host cannot infer from a different dictionary that the intended gate actually changed the computation.

The candidate source must implement the control. Prefer constructing all modules in both arms and gating outputs so configuration does not change initialization. Shared weights are retrained; even same-source controls do not automatically establish isolated attribution. The host records full validation metrics, an exploratory paired candidate-minus-control interval, actual control runtime and completion status. Negative differences favor the candidate.

One test trial spends at most **one candidate training plus one control training**. Deferring interactions spends no control training. A control failure preserves the completed main metrics and requests review, with no manufactured comparison. A main implementation may still be eligible for recipe-level promotion; `change_audit` and component-benefit attribution remain `unverified`.

The frozen protocol includes this control budget. Two interaction trials plus baseline can therefore require five full training runs. Old task/code fingerprints cannot resume under this contract.

## Editable native example

[examples/mixed_fm.py](../examples/mixed_fm.py) extends the existing embedding MLP with Harness-native PyTorch `NumericFieldEmbedding` and `GroupedFM`:

- Numeric values become `x_i * v_i`; missing numeric values are masked before multiplication.
- The categorical table is shared with the MLP. Three separate output terms describe category-category (`cc`), numeric-category (`nc`) and numeric-numeric (`nn`) relations.
- `enabled_pairs` gates those terms; `fm_scale` controls their residual scale. All modules are always constructed and called. Missing flags remain in the MLP in this example.

These are example configuration names, not a global strategy enumeration or new default seed. The Agent can implement other scopes, structures, higher-order products, gates, losses or fusion. Raw-field support, leakage checks and train-only preprocessing still matter. Existing available-field crosses do not require a human dataset refresh.

The Harness also contains an independent TensorFlow version of both primitives. CTREvo itself trains native PyTorch candidates only.

## Full-data verification

The [reproduction script](../experiments/explicit-interactions-20261006/verify.py) uses fixed proposals to verify source delivery, local composition history, strict execution checks, same-source controls, reflection and selection. It is **not a fresh DeepSeek search**, and does not measure whether an LLM chose the best priority. The test partition stays sealed.

```bash
python experiments/explicit-interactions-20261006/verify.py \
  --data /data/criteo/prepared --image ctrevo-cuda --venv "$PWD/.venv" \
  --output runs/explicit-interactions
```

Measured results and final test counts are recorded alongside the [verification report](../experiments/explicit-interactions-20261006/verification-report.json).

### Measured result — 2026-10-06

RTX 5090, PyTorch, seed 42, batch 8,192, one complete epoch per arm. Every arm trained on **36,672,493 rows** and evaluated **4,584,062 validation rows**; 4,584,062 test rows remained reserved. No sampling was used.

| Recipe | Validation LogLoss | Compared with | Paired difference (95% exploratory interval) |
| --- | ---: | --- | --- |
| Original seed MLP | 0.445336354 | Initial comparator | — |
| Same-source `cc` control, scale .05 | 0.445108141 | Control for mixed term | — |
| `cc + nc`, scale .05 | **0.444839234** | `cc` control | **−0.000268906 [−0.000330049, −0.000207764]** |
| `cc + nc + nn`, scale .05 | 0.444974778 | Same-source `cc + nc` control | **+0.000135544 [+0.000080420, +0.000190668]** |

The mixed term helped this recipe; adding numeric-numeric terms worsened it. Both trial implementations passed the bounded host execution check, and both controls completed. Strict selection retained `trial_001`; it rejected `trial_002` for no gain over the incumbent. This supports retaining scope-specific evidence instead of assuming that more interactions always help.

These are fixed-plan validation comparisons around the MLP, not new Agent-selected hypotheses or independent test improvements. They do not outperform the previously published Cross recipe's validation result (0.441451155). The `.05` residual scale and new shared initialization differ from the historical unscaled FM trial, so the historical FM failure is not a matched control here. Anonymous rows, one seed and adaptive-validation reuse limit the intervals; they do not correct unknown user clustering or selection. Whole-recipe control evidence does not establish the contribution of individual feature pairs.

The report records host-read/delivered source hashes, `.output_scales` and `.interaction_control` evidence IDs, actual runtime and source, paired results, local inheritance, reflection and promotion decisions. The unchanged configuration comparison was reproduced exactly during final verification.

### Checks

- [39 CTREvo regression tests](../experiments/explicit-interactions-20261006/ctrevo-explicit-tests.txt) passed on the GPU host, including same-source controls, no-op configuration rejection, failed-control preservation, bounded scales and probe transparency.
- [93 related Harness contract tests](../experiments/explicit-interactions-20261006/harness-contracts.txt) passed, including optional/required plans, malformed Agent JSON, deferred-plan history, reference delivery and existing engine/composition behavior.
- [Native source and PyTorch GPU tests](../experiments/explicit-interactions-20261006/ctrevo-explicit-model-tests.txt) passed. TensorFlow was not installed on this host; its test class was skipped. Its independent source is included, but this is not a dual-framework runtime pass.

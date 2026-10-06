# CTREvo design

Build an independent, Apache-2.0 CTR experiment host around unmodified ModelEvoHarness. The first full-data task uses Criteo Display Advertising (39 original fields), subject to verified download completeness. The Agent iterates actual PyTorch candidate code; model names do not constrain the search. Public English/Chinese documentation must distinguish tests, actual experiments and measured gains.

## Boundaries

- Harness is a Git submodule at the currently verified commit. Do not modify or update it without the user's confirmation. Reuse its engine, research validation, reference reading, horizontal composition and dataset-scoped journal.
- Host owns complete-data preparation, fixed train/validation/final partitions, preprocessing fit only on training, GPU-only model execution, objectives, scoring and resource limits.
- Candidate exports `build_model(schema, config)` returning a native PyTorch module accepting `(dense, categorical)` and producing one logit per row. It may export `training_loss(logits, labels)`; the host retains sample coverage, optimizer setup, training budget and evaluation.
- Agent proposes Python source plus bounded training settings. It receives current code, diagnostics, hypotheses and reflections; retry feedback preserves already read sources. Ordinary proposal/reflection use high; anomalous results trigger max review.
- Docker workers receive training data and evaluation features; validation/final labels remain on the host. Workers have no API credentials, network or writable evaluator source. Only selected native Harness model source is exposed.
- Validation LogLoss is the selection objective (minimize); also report AUC, Brier score and calibration summaries. Paired LogLoss intervals describe exploratory uncertainty, not post-selection proof. Final evaluation is separate, after freezing a candidate, and is not fed to Agent memory.
- Anonymous Criteo fields do not establish user/item/sequence semantics. Crosses, representations, parallel branches and loss changes remain open. Do not fabricate missing behavioral fields or a business causal explanation.

## Data protocol

Use every labeled row in the public release, preserving original order for an explicitly positional 80/10/10 split (not a verified chronological split). Unlabeled competition test rows cannot supply offline metrics. Every split row count, source hash and preprocessing choice is recorded. Feature hashing is fixed by field; missing categorical values have their own token. Dense scaling fits training statistics only. Missing indicators are derived inputs and reported separately from 39 original fields. No sample-size knob is provided for claimed full-data experiments.

## Success criteria

1. Reproducible full-data download/preparation with provenance and explicit field count.
2. Unmodified Harness drives proposal → GPU candidate evaluation → metrics → component reflection → next proposal.
3. Reports show what the Agent changed, whether it ran, and baseline/candidate metrics without inventing improvement.
4. Isolated public GitHub repository includes both READMEs, tests, source attribution, license and experiment reproduction instructions; raw data, keys and model weights stay outside Git.

# CTR host implementation plan

**Goal:** Connect a complete CTR task to the existing Harness and execute full-data GPU iterations.
**Architecture:** Streaming data preparation, native candidate interface, isolated GPU worker, host metrics/task adapter, provider adapter, CLI and public evidence.
**Spec:** [design.md](design.md)

1. Data and metrics: write tests for field width, missing values, stable hashing, train-only normalization, full row count and paired comparisons; implement preparation and independent scoring.
2. Execution: test candidate interface and invalid outputs; implement native seed model, GPU worker and Docker mounts that exclude validation labels and secrets.
3. Harness integration: test proposal validation/retry, source context, reflection and dataset identity; implement task/provider adapters without changing Harness.
4. CLI/documentation: add fetch/prepare/search/finalize commands, bilingual READMEs, CI and Apache-2.0; test metadata/CLI locally and tensor behavior on GPU.
5. Experiments: verify full release, train seed and Agent candidates on target GPU, retain reports and actual outcomes. Publish independent repository through gh.

Review focus: held-out leakage; stale dataset artifacts; positional split claims; candidate failure recovery; false attribution of joint architecture/training changes. No CPU model-experiment fallback.

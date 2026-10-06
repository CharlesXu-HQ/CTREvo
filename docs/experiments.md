# Experiment status

Implementation and interface tests are available. The full Criteo archive is downloading on the GPU host. No CTR improvement is claimed yet.

The intended first measured run uses the complete labeled release, an RTX 5090, one full epoch per candidate, a seed embedding MLP and two Agent-proposed code revisions. Results will be recorded only after execution, including failures and hypotheses that do not improve LogLoss.

## Contract verification — 2026-10-06

- Local: 11 tests, 10 passed and one CUDA-only test explicitly skipped.
- RTX 5090 host: all 11 tests passed, including native PyTorch forward/backward.
- Docker contract fixture: training, CUDA prediction and host scoring completed; all 16 fixture training rows consumed. This fixture is not a benchmark.
- Existing ModelEvoHarness submodule remains unchanged at `23c947ddc90b0a6ebafddfebad228021f1a54d5e`.
- DeepSeek model discovery succeeded and returned `deepseek-flash`; Agent-led dataset experiments have not yet finished.

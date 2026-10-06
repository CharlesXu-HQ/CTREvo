# Contributing

Keep the host evaluator independent from candidate code. Changes to preprocessing, splitting, metrics or training budget change the experiment identity and require a new run.

Run `python -m unittest discover -s tests`. GPU tests skip explicitly without CUDA; passing CPU tests is not evidence of a completed GPU experiment.

Use unit fixtures only to test contracts. Publish reproducible configurations and aggregate evidence, not credentials, raw datasets or large checkpoints. Describe changes to Agent authority or held-out data access explicitly.

ModelEvoHarness is a separate project. Propose changes upstream separately; do not silently modify the vendored submodule while evaluating CTREvo.

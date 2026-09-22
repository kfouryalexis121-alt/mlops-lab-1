# Lab 2 answers

## 1. Dependency-file changes

`pyproject.toml` now declares `mlflow`, `torch`, `torchvision`, and
`scikit-learn` as direct project dependencies. `uv.lock` expanded to record the
exact resolved versions, hashes, platform-specific wheels, and transitive
dependencies needed for a reproducible installation. This machine resolved the
CPU builds of PyTorch 2.14.0 and torchvision 0.29.0.

## 2. MLflow metadata and artifacts

`--backend-store-uri sqlite:///mlflow.db` tells MLflow where to keep structured
tracking metadata: experiments, runs, parameters, metric histories, tags,
statuses, and artifact locations. `--default-artifact-root ./mlruns` tells it
where to place artifact files for new experiments.

Metadata is small, structured information that MLflow queries and displays.
Artifacts are the actual files produced by a run, such as the serialized model,
its MLmodel definition, input examples, and environment files.

## 3. Why local MLflow output is ignored

`mlflow.db` and `mlruns/` are generated, machine-local, and change on every run.
The database is a mutable binary file, while model artifacts are comparatively
large. Tracking either with Git would create noisy history and repository
bloat. They also should not be put in DVC because they form MLflow's live
tracking store rather than a deliberate, immutable dataset or model snapshot;
MLflow already manages their identities and relationships.

## 4. Creating an experiment

The first `mlflow.set_experiment("food11")` call created the missing `food11`
experiment automatically and selected it for the subsequent run. It then
appeared in the MLflow UI.

## 5. Parameters versus metrics

A parameter is fixed configuration for a run, such as the learning rate or
batch size. It is logged once and does not need a step. A metric is a measured
numeric result that may change during training. Its `step` records where each
value belongs in the sequence, which lets MLflow store and graph the complete
per-epoch history.

## 6. Logged run contents and model location

The UI shows the run parameters, charts for `train_loss`, `val_loss`, and
`val_accuracy`, and the logged `model`. With MLflow 3.16, the winning run's
serialized PT2 model lives at:

`mlruns/2/models/m-e113c67e46814967a90386e3e3fd69a9/artifacts/data/model.pt2`

The model was also loaded back through its `runs:/.../model` URI and produced an
output tensor of shape `(1, 11)`.

## 7. Learning-rate comparison

For batch size 32, `0.0001` gave the best final validation accuracy at 73.08%.
Higher was not better: `0.001` finished at 48.36%, and `0.01` finished at
13.14%. The `0.001` run peaked at 58.12% in epoch 3 before dropping, while the
`0.0001` run improved steadily through epoch 5.

## 8. Parallel-coordinates pattern

Learning rate was the strongest pattern in these runs: validation accuracy rose
sharply as the learning rate decreased from `0.01` to `0.0001`. At learning
rate `0.001`, batch size 64 finished higher than batch size 32 (62.96% versus
48.36%). This is only one seeded run per setting, so it is evidence for this
experiment rather than a general rule about batch size.

## 9. Best run

| Learning rate | Batch size | Final validation accuracy | Test accuracy | Run ID |
|---:|---:|---:|---:|:---|
| 0.0001 | 32 | 73.08% | 76.28% | `f969a20016df4031b39cf615cf30c828` |
| 0.001 | 64 | 62.96% | 68.16% | `9613eaa40ce84370b88ac4d8bb37ed97` |
| 0.001 | 32 | 48.36% | 49.54% | `978ba6de62bf4f2595398857b063860f` |
| 0.01 | 32 | 13.14% | 15.69% | `8cdf8568a76a4c91a46927bcf2b7969f` |

The best run ID for the next lab is
`f969a20016df4031b39cf615cf30c828`.

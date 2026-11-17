# Lab 2 answers

## 1. Dependency-file changes

`pyproject.toml` adds `mlflow`, `torch`, `torchvision`, and `scikit-learn`.
`uv.lock` records exact versions and dependencies for reproducible installs,
including CPU builds of PyTorch 2.14.0 and torchvision 0.29.0 on this machine.

## 2. MLflow metadata and artifacts

`--backend-store-uri sqlite:///mlflow.db` stores tracking metadata, such as
runs, parameters, metrics, and tags. `--default-artifact-root ./mlruns` sets
the artifact location for new experiments. Artifacts are output files, such
as saved models, input examples, and environment files.

## 3. Why local MLflow output is ignored

`mlflow.db` and `mlruns/` change with each run and can contain large files,
creating noisy Git history. We also exclude them from DVC because they are
MLflow's live tracking store, rather than fixed dataset or model snapshots.

## 4. Creating an experiment

`mlflow.set_experiment("food11")` automatically created the missing experiment,
selected it for the run, and made it visible in the MLflow UI.

## 5. Parameters versus metrics

Parameters are fixed run settings, such as learning rate and batch size.
Metrics are measured values that can change during training. A metric's
`step` identifies its epoch or iteration so MLflow can plot its history.

## 6. Logged run contents and model location

The UI shows parameters, `train_loss`, `val_loss`, `val_accuracy`, and the
logged model. The winning run's model is stored at:

`mlruns/2/models/m-e113c67e46814967a90386e3e3fd69a9/artifacts/data/model.pt2`

Loading it through `runs:/.../model` produced an output tensor of shape `(1, 11)`.

## 7. Learning-rate comparison

With batch size 32, learning rate `0.0001` achieved the best final validation
accuracy (73.08%), compared with 48.36% for `0.001` and 13.14% for `0.01`.
The `0.0001` run improved steadily; `0.001` peaked at 58.12% in epoch 3.

## 8. Parallel-coordinates pattern

Lower learning rates gave better validation accuracy in these runs. At
`0.001`, batch size 64 outperformed 32 (62.96% versus 48.36%). With only one
seeded run per setting, this is not a general rule about batch size.

## 9. Best run

| Learning rate | Batch size | Final validation accuracy | Test accuracy | Run ID |
|---:|---:|---:|---:|:---|
| 0.0001 | 32 | 73.08% | 76.28% | `f969a20016df4031b39cf615cf30c828` |
| 0.001 | 64 | 62.96% | 68.16% | `9613eaa40ce84370b88ac4d8bb37ed97` |
| 0.001 | 32 | 48.36% | 49.54% | `978ba6de62bf4f2595398857b063860f` |
| 0.01 | 32 | 13.14% | 15.69% | `8cdf8568a76a4c91a46927bcf2b7969f` |

The best run for the next lab is `f969a20016df4031b39cf615cf30c828`.

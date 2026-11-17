# Lab 5 answers

## 1. Why unit tests use a fake model

A real model introduces unrelated failures: MLflow may be offline, credentials
may expire, artifacts may be missing, or the alias may change. Unit tests should
check code deterministically without a server, GPU, or downloaded dataset.

## 2. What the fake-model test proves

It checks image decoding, preprocessing shape/type, class mapping, softmax,
and the response JSON. It does not measure classification accuracy or prove
that the real model can be downloaded. Our fake returns log probabilities so
the API's softmax produces 90% confidence for Bread.

## 3. Detecting duplicates

The content-hash check detected the deliberate training/evaluation duplicate.
Unlike a corrupt image, leakage can silently inflate evaluation scores and
make an overfitted model look ready to deploy.

The original mini dataset also contained 43 cross-split duplicate copies.
`prepare_lab5_data.py` creates a separate deduplicated copy under
`data_local/food11_lab5`, preserving evaluation images and the original data.
The clean copy passed all five checks; adding one duplicate made only the
leakage check fail, and the probe was then removed.

## 4. Where the data contract runs

Run it immediately after data preparation/DVC restoration, and before training
or promotion in an environment that can access the dataset. Training now
refuses to start when validation fails. Cloud CI tests the validator on small
synthetic datasets; it does not claim to validate unavailable DVC data.

## 5. Job dependencies

If `test` fails, `smoke` is skipped because it declares `needs: test`.
Without that dependency, both jobs could run independently and waste a build
on code whose tests already fail. Publishing depends on successful smoke tests.

## 6. Run time and caching

The workflow caches uv packages and Docker build layers. A repeated run can
reuse dependency downloads and unchanged image layers. Measured GitHub run
times will be recorded after the cold and warm runs complete.

## 7. Commit SHA versus latest

A SHA tag identifies the source commit. If machine A pulls `latest` before a
new push and machine B pulls afterward, they can run different code under the
same tag. Use an image digest when byte-for-byte identity must be enforced.
This workflow publishes the exact image tested by `smoke`, rather than
rebuilding it in the publishing job.

## 8. Token permissions

Tests and builds only need repository read access. Only the publishing job
gets `packages: write`. A broadly privileged compromised step could modify
repository contents, releases, or settings instead of just publishing images.

## 9. Protected pull requests

The protection experiment requires a pull request and successful `test` and
`smoke` checks on `main`. A failing test should block merging and skip smoke.
The actual failing/fixed pull-request results will be recorded after the test.

## 10. Candidate promotion

The gate compares candidate accuracy with `p + 2 * sqrt(p * (1-p) / n)`, where
`p` is champion accuracy and `n` is the evaluation size. Passing candidates
become `champion`; others become `challenger`. Both accuracies, the threshold,
sample size, previous champion, and decision are stored as version tags.
Measured results will be added after both training runs finish.

## 11. Automating model promotion

Run the promotion command after training on a local/self-hosted runner with
MLflow access, or host MLflow on a reachable service such as DagsHub and supply
credentials to the workflow. Promotion does not automatically restart the API;
that remains a separate deployment action.

## 12. Pipeline

```mermaid
flowchart TD
    A[Human: git push or pull request] --> B[Automatic: locked dependencies, lint, unit tests]
    B --> C[Automatic: Docker build and SKIP_MODEL smoke test]
    C --> D{Push to main?}
    D -->|Yes| E[Automatic: publish tested image to GHCR with commit SHA]
    D -->|PR| F[Human: merge after required checks pass]
    F --> A
    E --> G[Human: pull image and deploy container]
    H[Human: prepare data and start training] --> I[Automatic: data contract and logged training run]
    I --> J[Human: register candidate and invoke promotion script]
    J --> K[Automatic: quality gate selects champion or challenger]
    K --> L[Human: restart inference]
    G --> M[Startup: resolve champion and fetch model]
    L --> M
```

The image's code version is chosen by its commit SHA/digest. The registry's
`champion` alias selects the model version when inference starts. CI never
trains the production model and does not deploy to the laptop automatically.

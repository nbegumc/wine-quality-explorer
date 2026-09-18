# Wine Quality Explorer

[![CI](https://github.com/nbegumc/wine-quality-explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/nbegumc/wine-quality-explorer/actions/workflows/ci.yml)

**Live dashboard:** https://nbegumc.github.io/wine-quality-explorer/

Can laboratory measurements tell you how a red wine will score — and how should your belief change when you only know *some* of them? This project answers with a Bayesian network you can query with any subset of eleven measurements, evaluated honestly against simpler models on the UCI red Vinho Verde data.

The two models have different jobs:

- **Logistic regression is the prediction reference.** Validation selected it; use it when all eleven measurements are known (60.0% holdout accuracy).
- **The Bayesian network is the reasoning model.** It answers with partial evidence, returns a full distribution over scores 3–8, and exposes a dependency graph whose stability you can inspect. It predicts less accurately (56.6%) and the dashboard says so.

## What the dashboard does

- **Ask the network** — tick the measurements you know, read the probability of each score, and see the logistic-regression reference beside it once all eleven are given.
- **Compare models** — seven prespecified candidates on the same folds and one untouched holdout: accuracy, macro-F1, recall by class, confusion matrix, calibration.
- **Explore the network** — the learned graph, with each arrow's bootstrap stability and direction share.
- **Data & methodology** — the split, the selection rule, and the design decisions.

Inference runs in the browser from exported probability tables; no server logic, no retraining.

## Run it

Serve the dashboard (Python only):

```bash
uv run scripts/serve.py          # http://localhost:8000
```

or with Docker:

```bash
docker compose up
```

Reproduce the experiment in the reference environment and check it against the committed results:

```bash
docker compose run --rm experiment
uv run python tests/compare_results.py <(git show HEAD:app/results.json) app/results.json
```

Run the tests:

```bash
uv run python -m unittest discover -s tests -v
node tests/test_inference.mjs    # browser inference vs Python, 41 queries
```

Without Docker, `uv run scripts/train.py`, `scripts/bootstrap_structure.py`, and `scripts/export_reference.py` rerun the three stages natively — but only the `linux/amd64` image reproduces the network results exactly; on arm64, pyAgrum's structure search breaks a near-tie differently and selects another network (`docs/project_setup.md`). Rebuild the guide's PDF and Word editions with `uv run --with python-docx --with reportlab scripts/build_guide.py`.

## Results

One frozen holdout of 320 wines; candidates selected by mean five-fold validation macro-F1 before the holdout was evaluated.

| Model | Validation macro-F1 | Holdout accuracy | Holdout macro-F1 |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.100 | 42.5% | 0.099 |
| **Logistic regression** (prediction reference) | **0.289** | 60.0% | 0.397 |
| Random forest | 0.281 | 62.8% | 0.308 |
| BN · BIC | 0.256 | 54.1% | 0.244 |
| **BN · AIC** (reasoning model) | 0.267 | 56.6% | 0.277 |
| BN · BIC + prior arcs | 0.264 | 54.1% | 0.265 |
| BN · AIC + prior arcs | 0.256 | 57.5% | 0.278 |

Random forest has the best holdout accuracy but was not selected by the prespecified rule; picking it afterwards would be selecting on the test score. The network's gap is structural: measurements are cut into three bins and the network models all twelve variables jointly. In an exploratory check on the training folds, training logistic regression on the same bins cost it about three accuracy points — roughly half the gap. What the network buys is inference from partial evidence, which the other models cannot do without imputation.

Bootstrap arrow stability (1,000 resamples): 12 of the 28 learned arrows appear in at least 85% of resampled graphs, 9 in fewer than half, and several stable arrows have a direction share near 50% — the data supports the dependency but not the arrowhead.

## How the experiment works

1. Load the 1,599-row CSV: 11 measurements, scores 3–8.
2. Group identical measurement rows so duplicates never cross a split.
3. Freeze one stratified group fold (seed 123) as the holdout: 1,279 training rows, 320 held out.
4. Compare seven candidates on five group-aware validation folds inside training. Each Bayesian network relearns quantile cut points, structure (greedy hill climbing, AIC or BIC, at most three parents), and BDeu-smoothed tables per fold; seven prespecified arcs are tested as a constraint condition.
5. Select the prediction reference and the reasoning network by mean validation macro-F1.
6. Fit the frozen candidates on all training rows and evaluate the holdout once.
7. Export the training-only network's tables and the reference model's coefficients for the browser; relearn the network on bootstrap resamples to score each arrow's stability.

`docs/METHODOLOGY.md` explains the reasoning behind each step; `docs/BEGINNER_GUIDE.md` teaches the project from probability upward (also as [PDF](app/wine-quality-learning-guide.pdf) and [Word](app/wine-quality-learning-guide.docx), both generated from the Markdown).

## Layout

| Path | Purpose |
| --- | --- |
| `src/wine_quality/model.py` | Binning, structure learning, exact inference, table export, bootstrap arc strength |
| `scripts/train.py` | The prespecified experiment: splits, candidates, selection, holdout, exports |
| `scripts/bootstrap_structure.py`, `scripts/export_reference.py` | Arrow stability; reference-model parameters for the browser |
| `scripts/serve.py`, `scripts/build_guide.py`, `scripts/package_download.py` | Local server; guide PDF/Word; source archive |
| `app/` | The dashboard (`index.html`, `app.mjs`, `inference.mjs`) and its data contract `results.json` |
| `data/` | Source CSV, exported network (`.bif`), holdout predictions |
| `tests/` | Split integrity, preprocessing, probabilities, browser parity, reproduction check |
| `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml` | Pinned amd64 reference environment; tests, reproduction check, Pages deploy |

## Data and licenses

Cortez, Cerdeira, Almeida, Matos, and Reis (2009), [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), UCI Machine Learning Repository, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), **CC BY 4.0**. The CSV is included unchanged; its SHA-256 is recorded in `results.json`. Quality scores are sensory ratings of Portuguese red Vinho Verde wines: no claim is made about other wines, about causation, or about reliably identifying the rare extreme scores.

Code is released under the [MIT License](LICENSE).

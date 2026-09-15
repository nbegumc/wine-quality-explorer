# Wine Quality Explorer

[![CI](https://github.com/nbegumc/wine-quality-explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/nbegumc/wine-quality-explorer/actions/workflows/ci.yml)

**Live dashboard:** https://nbegumc.github.io/wine-quality-explorer/

A Python rebuild of **Applying Bayesian Networks to Wine Quality Prediction**, an R project completed during university studies. The rebuild keeps the original research question and makes the evaluation reproducible, the predictions interactive, and the limitations visible.

The two models have different jobs. **Logistic regression is the prediction reference**: validation selected it, and it is what to use when all eleven measurements are known. **The Bayesian network is the reasoning model**: it answers with any subset of measurements, returns a full distribution over scores, and exposes a dependency structure whose stability can be inspected. It predicts less accurately than the reference, and the dashboard says so.

## Try the dashboard

You only need Python to explore the already-trained results:

```bash
python serve.py
```

Open **http://localhost:8000**. Keep the terminal open while using the app. Stop it with Ctrl+C. No account, API key, or GitHub connection is required.

With Docker instead of a local Python:

```bash
docker compose up
```

The same address serves the dashboard from the project's pinned image. The hosted copy at the top of this page is deployed from `main` by GitHub Actions.

The dashboard contains:

- The network’s quality distribution under partially observed measurements, with the logistic-regression reference shown beside it whenever all eleven measurements are given.
- Held-out examples whose actual quality can be compared with both models.
- Seven-model comparisons, confusion matrices, per-class recall, and a calibration plot.
- A clickable Bayesian network with bootstrap arrow stability, and a record of the methodology changes.

The interface evaluates conditional probabilities from exported Python-trained tables directly in the browser. It does not run training again when a control changes. All model data is included. Optional web fonts have system-font fallbacks.

## Learn the project from the beginning

The [beginner learning guide](docs/BEGINNER_GUIDE.md) explains the objective, probability and Bayes' rule using small counting examples, network structure and tables, every experiment step, the actual results, and a real failed prediction. It includes dashboard exercises, a code map, a glossary, and references. No probability background is assumed.

Download the same guide as a [PDF](app/wine-quality-learning-guide.pdf) or an [editable Word document](app/wine-quality-learning-guide.docx). Both are generated from the Markdown by `build_guide.py` and are linked from the dashboard. This is the second edition, written for the explorer that pairs the network with a prediction reference; the first edition is kept as [`docs/BEGINNER_GUIDE_old.md`](docs/BEGINNER_GUIDE_old.md) with its own [PDF](app/wine-quality-learning-guide_old.pdf) and [Word](app/wine-quality-learning-guide_old.docx) files. The shorter [R audit and implementation notes](docs/LEARNING_GUIDE.md) remain available for the specific corrections to the original project.

## Reproduce the experiment

Use Python 3.11 or newer. The project is managed with [uv](https://docs.astral.sh/uv/), which creates the environment, installs the locked dependencies, and fetches the right Python itself:

```bash
uv run train.py
uv run bootstrap_structure.py
uv run export_reference.py
uv run python -m unittest discover -s tests -v
uv run serve.py
```

After editing `docs/BEGINNER_GUIDE.md`, regenerate its PDF and Word editions with `uv run --with python-docx --with reportlab build_guide.py`; the two libraries are fetched for that run only and are not project dependencies.

To reproduce the published numbers exactly, run the experiment in the reference environment, the pinned `linux/amd64` Docker image:

```bash
docker compose run --rm experiment
uv run python tests/compare_results.py app/results.json <(git show HEAD:app/results.json)
```

The container rewrites `app/results.json`, `data/selected-network.bif`, and `data/holdout-predictions.csv` in the checkout; the comparison confirms every field matches the committed file (floats within 1e-9). CI performs the same check on every push. A native run on Apple silicon does **not** reproduce the network results: pyAgrum's structure search breaks near-ties differently on arm64 and selects a different network. `docs/project_setup.md` records the finding.

`train.py` runs the experiment. `bootstrap_structure.py` then relearns the selected network's recipe on 1,000 group-bootstrap resamples of the recorded training partition (about half a minute) and adds each arrow's stability to `app/results.json` without changing any other result. `export_reference.py` refits the validation-selected prediction model on the recorded training partition, checks that it reproduces the recorded holdout metrics, and adds its parameters for the browser. The dashboard works without either step; it then shows no stability information and no reference prediction.

No activation step is needed: each `uv run` syncs `.venv` against `uv.lock` first. Dependencies are declared in `pyproject.toml`; `uv.lock` pins every transitive package for all platforms.

Without uv, create an environment and install the exported pins instead:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate        # macOS/Linux
python -m pip install -r requirements.txt
python -m pip install -e .
python train.py
python bootstrap_structure.py
python export_reference.py
python -m unittest discover -s tests -v
python serve.py
```

`requirements.txt` is generated from the lockfile with `uv export --no-hashes --no-emit-project -o requirements.txt`; edit `pyproject.toml` and re-export rather than editing it by hand. The exact versions actually used for the published results are also recorded in `app/results.json`.

The [project setup notes](docs/project_setup.md) record how the environment is managed, the commands for changing dependencies, and a known sensitivity in model selection.

If Node.js is available, independently check the browser's exact-inference implementation:

```bash
node tests/test_inference.mjs
```

This compares 41 browser queries with Python inference, including no evidence, partial evidence, all measurements, and every numerical bin boundary. CI runs it on every push, together with the Python tests and the container reproduction check.

## What the experiment actually does

1. Load the original red-wine CSV: 1,599 rows, 11 predictors, scores 3–8.
2. Group identical predictor rows. The 240 repeated rows are retained, but copies cannot cross training, validation, or holdout boundaries.
3. Reserve the first fold of a five-fold `StratifiedGroupKFold` with seed 123: 1,279 training rows and 320 holdout rows.
4. Within training, compare seven prespecified models on the same five group-aware validation folds. Each Bayesian network relearns quantile cut points, graph structure, and parameters within its own training fold. Logistic regression fits its scaler within the fold.
5. Select the overall model and the Bayesian network using **mean validation macro-F1**, before evaluating the holdout.
6. Fit each prespecified model on the training partition and evaluate once on the shared holdout. No model is changed in response to these scores.
7. Export the chosen Bayesian network fitted on training data only. This is the same model evaluated in the results, not a later refit on all observations.

### Models

- Majority-class baseline (empirical class-prior probabilities).
- Standardized multinomial logistic regression, `C=1`, maximum 3,000 iterations.
- Random forest, 300 trees, minimum leaf size 2, seed 123.
- Greedy hill-climbing Bayesian networks with AIC or BIC, each with or without the original seven mandatory arcs.

All Bayesian networks have a maximum of three parents per node. Structure scores are applied without an added prior. Parameters are then estimated with a BDeu prior of equivalent sample size 5. Predictions use exact inference conditioned on all provided measurements.

## Recorded results

These describe **one frozen holdout**, not guaranteed performance on other wines.

| Model | Validation macro-F1 | Holdout accuracy | Holdout macro-F1 |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.100 | 42.5% | 0.099 |
| Logistic regression | 0.289 | 60.0% | 0.397 |
| Random forest | 0.281 | 62.8% | 0.308 |
| BN · BIC | 0.256 | 54.1% | 0.244 |
| BN · AIC | 0.267 | 56.6% | 0.277 |
| BN · BIC + original priors | 0.264 | 54.1% | 0.265 |
| BN · AIC + original priors | 0.256 | 57.5% | 0.278 |

Validation selected **logistic regression** overall and **BN · AIC without forced priors** among the networks. Logistic regression is therefore the prediction reference, and the network is the reasoning model of the explorer. Random forest has the highest holdout accuracy but was not selected by the prespecified rule; choosing it afterwards would mean selecting on the test score.

The network’s gap to the reference is structural rather than a defect: every measurement is cut into three bins before learning, and the network models the joint distribution of all twelve variables instead of the quality boundary alone. In an exploratory five-fold check on the training rows (not part of the recorded experiment), training logistic regression on the same three bins lowered its validation accuracy from 0.599 to 0.571, about half of the network’s gap; finer bins recovered accuracy but starved the rare classes. What the network offers instead is inference from partial evidence, which the reference models cannot provide without imputation.

The reported accuracy intervals use 500 percentile bootstrap samples of holdout measurement groups. They are conditional on the trained model and this split. They do not include all training, model-selection, or population uncertainty. Rare-class estimates are particularly unstable. A lower mean absolute error is better; macro-F1 weights each of the six classes equally. Quality is treated as nominal during fitting, while ordinal errors are additionally reported.

## Differences from the R version

Read `docs/LEARNING_GUIDE.md` for explanations tied to the original mistakes.

This is an intentional first rebuild, not a line-for-line replication:

- Train-only quantile discretization replaces Hartemink's pairwise information-preserving discretization.
- pyAgrum greedy hill climbing replaces the R comparison of Grow–Shrink, hill climbing, and MMHC.
- Maximum indegree is fixed at three; exact inference replaces prediction from parents alone.
- The AIC/BIC evaluation bug is corrected.
- Cross-validation repeats the complete learning process.
- The original's graph bootstrap is reproduced only as a diagnostic: `bootstrap_structure.py` relearns the selected recipe on 1,000 resamples of training measurement groups and records how often each arrow appears and in which direction. The original instead kept arrows found in at least 85% of resamples as its model; no averaged network is used for prediction here. The accuracy interval bootstrap is a different operation and must not be presented as structure averaging.
- The original prior edges are a comparison condition only. Their chemical or causal directions are not verified by this project.

## Project map

| File | Purpose |
| --- | --- |
| `src/wine_quality/model.py` | Binning, network learning, exact inference, and table export |
| `train.py` | Grouped evaluation, selection, results, and prediction exports |
| `bootstrap_structure.py` | Arrow stability of the selected network from bootstrap relearning |
| `export_reference.py` | Parameters of the validation-selected prediction model for the browser |
| `build_guide.py`, `docs/guide-template.docx` | Generate the guide's PDF and Word editions from `docs/BEGINNER_GUIDE.md` |
| `serve.py` | One-command local dashboard (`HOST`/`PORT` from the environment) |
| `Dockerfile`, `compose.yaml` | Pinned amd64 reference environment: `dashboard` and `experiment` services |
| `.github/workflows/ci.yml` | Tests, browser parity, container reproduction check, Pages deployment |
| `tests/compare_results.py` | Field-by-field comparison of a regenerated `results.json` with the committed one |
| `app/index.html`, `styles.css`, `app.mjs` | Interactive browser interface |
| `app/inference.mjs` | Exact variable elimination over exported probability tables |
| `app/results.json` | Metrics, splits, software versions, cut points, graph, tables, arrow stability, and the reference model |
| `data/winequality-red.csv` | Original attributed dataset |
| `data/selected-network.bif` | Interoperable trained Bayesian network |
| `data/holdout-predictions.csv` | Row-level predictions and probabilities |
| `tests/` | Checks for split integrity, preprocessing, probabilities, and browser parity |
| `original/` | Unmodified original R script and README |

## Data and limitations

Source: Cortez, Cerdeira, Almeida, Matos, and Reis (2009), [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), UCI Machine Learning Repository, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T). Dataset license: **CC BY 4.0**. The CSV is included unchanged; its SHA-256 is recorded with the experiment. Quality labels are sensory assessments.

The dataset represents Portuguese red Vinho Verde wines. No general claim about all wines, experimental causation, ingredient optimization, or dependable identification of rare extreme-quality wines is established. Grouping identical measurements reduces duplicate leakage, but without batch or producer identities it cannot prevent every form of dependence between samples.

The original university project remains the historical reference. The Python rebuild was developed with AI assistance; the learning guide and tests are provided so its decisions and calculations can be inspected and understood.

No new software license is selected on the author's behalf. The dataset's license is separate from the repository's code licensing.

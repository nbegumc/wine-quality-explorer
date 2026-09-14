# Wine Quality Explorer

A Python rebuild of **Applying Bayesian Networks to Wine Quality Prediction**, an R project completed during university studies. The rebuild keeps the original research question and makes the evaluation reproducible, the predictions interactive, and the limitations visible.

## Try the dashboard

You only need Python to explore the already-trained results:

```bash
python serve.py
```

Open **http://localhost:8000**. Keep the terminal open while using the app. Stop it with Ctrl+C. No account, API key, or GitHub connection is required.

The dashboard contains:

- Quality probabilities with partially observed measurements.
- Held-out examples whose actual quality can be compared with predictions.
- Seven-model comparisons, confusion matrices, per-class recall, and a calibration plot.
- A clickable Bayesian network and a record of the methodology changes.

The interface evaluates conditional probabilities from exported Python-trained tables directly in the browser. It does not run training again when a control changes. All model data is included. Optional web fonts have system-font fallbacks.

## Learn the project from the beginning

The [beginner learning guide](docs/BEGINNER_GUIDE.md) explains the objective, probability and Bayes' rule using small counting examples, network structure and tables, every experiment step, the actual results, and a real failed prediction. It includes dashboard exercises, a code map, a glossary, and references. No probability background is assumed.

Download the same guide as a [PDF](dist/wine-quality-learning-guide.pdf) or an [editable Word document](dist/wine-quality-learning-guide.docx). Both are also linked from the dashboard. The shorter [R audit and implementation notes](docs/LEARNING_GUIDE.md) remain available for the specific corrections to the original project.

## Reproduce the experiment

Use Python 3.11 or newer. Create an isolated environment and install dependencies:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
python serve.py
```

The exact versions actually used are also recorded in `dist/results.json`. All direct modelling dependencies are pinned. This is not a complete transitive lockfile; platform-specific dependencies may differ.

If Node.js is available, independently check the browser's exact-inference implementation:

```bash
node tests/test_inference.mjs
```

This compares 41 browser queries with Python inference, including no evidence, partial evidence, all measurements, and every numerical bin boundary.

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

Validation selected **logistic regression** overall and **BN · AIC without forced priors** among the networks. The explorer shows the latter because the research question concerns Bayesian networks. The comparison explicitly retains stronger alternatives.

The reported accuracy intervals use 500 percentile bootstrap samples of holdout measurement groups. They are conditional on the trained model and this split. They do not include all training, model-selection, or population uncertainty. Rare-class estimates are particularly unstable. A lower mean absolute error is better; macro-F1 weights each of the six classes equally. Quality is treated as nominal during fitting, while ordinal errors are additionally reported.

## Differences from the R version

Read `docs/LEARNING_GUIDE.md` for explanations tied to the original mistakes.

This is an intentional first rebuild, not a line-for-line replication:

- Train-only quantile discretization replaces Hartemink's pairwise information-preserving discretization.
- pyAgrum greedy hill climbing replaces the R comparison of Grow–Shrink, hill climbing, and MMHC.
- Maximum indegree is fixed at three; exact inference replaces prediction from parents alone.
- The AIC/BIC evaluation bug is corrected.
- Cross-validation repeats the complete learning process.
- Graph bootstrap averaging from the original is not yet reproduced. The accuracy interval bootstrap is a different operation and must not be presented as structure averaging.
- The original prior edges are a comparison condition only. Their chemical or causal directions are not verified by this project.

## Project map

| File | Purpose |
| --- | --- |
| `src/wine_quality/model.py` | Binning, network learning, exact inference, and table export |
| `train.py` | Grouped evaluation, selection, results, and prediction exports |
| `serve.py` | One-command local dashboard |
| `dist/index.html`, `styles.css`, `app.mjs` | Interactive browser interface |
| `dist/inference.mjs` | Exact variable elimination over exported probability tables |
| `dist/results.json` | Metrics, splits, software versions, cut points, graph, and tables |
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

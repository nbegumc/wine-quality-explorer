# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment is managed with uv (`pyproject.toml` + `uv.lock`, Python 3.12 via `.python-version`). `uv run` syncs `.venv` first; no activation step.

```bash
uv run train.py                                         # full experiment; regenerates app/results.json, data/selected-network.bif, data/holdout-predictions.csv
uv run python -m unittest discover -s tests -v          # Python tests (stdlib unittest, not pytest)
uv run python -m unittest tests.test_model.ModelTests.test_cutpoints_are_training_only -v   # single test
node tests/test_inference.mjs                           # browser-inference parity vs Python (41 saved queries, tol 1e-10)
uv run serve.py                                         # dashboard at http://localhost:8000 (serves app/ statically)
uv run package_download.py                              # builds app/wine-quality-python.zip
docker compose up                                       # dashboard from the pinned amd64 image
docker compose run --rm experiment                      # regenerate artifacts in the reference environment
uv run --with python-docx --with reportlab build_guide.py   # guide PDF/DOCX from docs/BEGINNER_GUIDE.md
```

Dependency changes: edit via `uv add <pkg>==<ver>`, then regenerate the pip fallback with `uv export --no-hashes --no-emit-project -o requirements.txt`. Never hand-edit `requirements.txt`.

## Architecture

Two halves connected by one JSON contract (`app/results.json`):

**Python side (training, offline)**
- `src/wine_quality/model.py` — `QuantileBins` (train-only tertile cut points → `low/medium/high`), `WineBN` (pyAgrum greedy hill-climbing structure learning with AIC/BIC, max indegree 3, then BDeu(5) parameter fit; `query`/`predict_proba` use `LazyPropagation` exact inference), and `WineBN.export()` which flattens every CPT in explicit C-order with `[node] + sorted(parents)` variable order.
- `train.py` — the prespecified experiment. Groups identical predictor rows (`hash_pandas_object`) so duplicates never cross splits; first fold of `StratifiedGroupKFold(5, seed=123)` is the frozen holdout; seven `CANDIDATES` compared on inner five-fold group CV; winners chosen by **mean validation macro-F1 before** any holdout evaluation. The exported BN is the training-only fit, not a refit on all data.

**Browser side (inference, no server logic)**
- `app/inference.mjs` — variable elimination over the exported factors. Must stay numerically identical to pyAgrum; `tests/test_inference.mjs` checks this against `reference_queries` that `train.py` writes into `results.json`.
- `app/app.mjs` — dashboard UI; reads `results.json` only.

**Contract invariants** — if you change any of these on one side you must change the other and rerun both test suites:
- Factor variable order (`[name] + sorted(parents)`), C-order flattening, sizes (`quality`=6, else 3).
- Discretization rule: `searchsorted(cuts, value, side="right")` — a value equal to a cut point goes to the upper bin. Python and JS both implement this; boundary cases are tested in both.
- `FEATURES` order, `CLASSES = 3..8`, `STATES = ["low","medium","high"]`.

## Things to know before changing anything

- `tests/test_model.py` is a regression suite against the **committed artifacts** (`app/results.json`, `data/selected-network.bif`). It reconstructs the model from those files and checks split integrity, train-only cut points, and prediction parity. Running `train.py` overwrites those artifacts; if outputs change, tests may still pass but the README results table and docs go stale.
- Model selection is numerically fragile: `BN · AIC` vs `BN · BIC + original priors` differ by ~0.005 macro-F1, and pyAgrum's hill climbing breaks the tie differently on arm64, so a native run on Apple silicon flips the exported network. The reference environment is the amd64 Docker image; regenerate artifacts only with `docker compose run --rm experiment` and compare with `python tests/compare_results.py app/results.json <regenerated>`. See `docs/project_setup.md` ("Reproducibility finding").
- The methodology is deliberately prespecified. Do not tune anything against holdout scores, change `SEED`, alter the candidate list, or move selection after holdout evaluation — the README and `docs/LEARNING_GUIDE.md` document these as corrections to the original R project in `original/`.
- `ORIGINAL_PRIORS` arcs are an experimental comparison condition only, not verified causal claims; keep that framing in any docs or UI text.
- `WineBN.fit` writes a temporary CSV under `Path.cwd()` (prefix `.wine-training-`, gitignored) because pyAgrum's `BNLearner` needs a file path.
- `docs/BEGINNER_GUIDE.md`, `docs/LEARNING_GUIDE.md`, and the PDF/DOCX in `app/` are user-facing teaching material that quote specific numbers from the results table; regenerating results means updating them.

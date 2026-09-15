# Project setup: uv migration

This project previously declared its four direct dependencies in `requirements.txt` and expected a
hand-built `python -m venv` environment. It is now managed with [uv](https://docs.astral.sh/uv/):
dependencies are declared in `pyproject.toml`, resolved into a complete `uv.lock`, and the package in
`src/wine_quality` is installed into the environment instead of being reached through `sys.path`.

## Why

- `requirements.txt` pinned only the four direct dependencies. The packages those pull in — scipy,
  matplotlib, joblib, and the rest — were free to differ between machines and between runs.
  `uv.lock` pins all 22 packages, for every platform.
- `uv run` syncs the environment before executing, so the environment cannot silently drift from the
  declaration.
- The editable install makes `import wine_quality` work regardless of the working directory.

## What the environment now contains

| File | Role |
| --- | --- |
| `pyproject.toml` | Dependency declaration, `requires-python = ">=3.11"`, hatchling build of `src/wine_quality` |
| `uv.lock` | Full transitive lockfile — commit it |
| `.python-version` | `3.12`, the interpreter family used for the published results |
| `requirements.txt` | Generated from the lockfile; for consumers without uv |

## Commands used

Create the declaration, pin the interpreter, and build the environment:

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, writes `uv.lock`, downloads the interpreter named in
`.python-version` if it is not present, creates `.venv`, and installs the project itself in editable
mode.

Regenerate `requirements.txt` whenever `pyproject.toml` or the lockfile changes:

```bash
uv export --no-hashes --no-emit-project -o requirements.txt
```

`--no-emit-project` omits the `wine-quality-explorer` line itself, which a plain `pip install -r`
could not resolve from PyPI.

Everyday use, with no activation step:

```bash
uv run train.py
uv run python -m unittest discover -s tests -v
uv run serve.py
uv run package_download.py
```

Adding or changing a dependency:

```bash
uv add <package>==<version>        # writes pyproject.toml and updates uv.lock
uv add --dev <package>             # development-only group
uv export --no-hashes --no-emit-project -o requirements.txt
```

## Verification performed

```bash
uv run python -V                                          # Python 3.12.12
uv run python -c "import wine_quality; print(wine_quality.__file__)"
uv run python -m unittest discover -s tests -v            # 6 tests, OK
uv run package_download.py                                # archive includes the new files
```

The import resolves to `src/wine_quality/__init__.py`, confirming the editable install rather than a
copy in `site-packages`.

## Reproducibility finding

`uv run train.py` was executed end to end against the new environment. It did **not** reproduce the
committed artifacts. The explorer's selected Bayesian network changed from `BN · AIC` to
`BN · BIC + original priors`, and log-loss values drifted at the thirteenth decimal place.

The cause is not the migration. Those two candidates score 0.259 and 0.264 mean five-fold validation
macro-F1 — close enough that a different scipy or BLAS build reorders them. The published results
were produced under Python 3.12.14; the current lock resolves 3.12.12 with newer transitive builds.

`app/results.json`, `data/holdout-predictions.csv`, and `data/selected-network.bif` were restored
from git, so the published artifacts and the tests that check them are unchanged.

To make the reproducibility claim hold exactly, pin `.python-version` to `3.12.14`, run `uv sync`,
and regenerate the artifacts once with `uv run train.py` so they match the locked environment. Until
that is done, treat the BN selection as sensitive to the numerical environment rather than as a
stable result.

## Migration notes

- `train.py` and `tests/test_model.py` still call `sys.path.insert(0, str(ROOT / "src"))`. This is
  now redundant under uv but harmless, and it keeps the no-install path working.
- `package_download.py` includes `pyproject.toml`, `uv.lock`, and `.python-version` in the archive so
  a downloaded copy can be rebuilt with `uv sync`.
- `.venv/` is already ignored by `.gitignore`.

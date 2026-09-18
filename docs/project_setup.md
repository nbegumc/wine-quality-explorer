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
uv run scripts/train.py
uv run python -m unittest discover -s tests -v
uv run scripts/serve.py
uv run scripts/package_download.py
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
uv run scripts/package_download.py                                # archive includes the new files
```

The import resolves to `src/wine_quality/__init__.py`, confirming the editable install rather than a
copy in `site-packages`.

## Reproducibility finding

The committed artifacts are reproduced exactly by the Docker image (`Dockerfile`, `compose.yaml`)
when it runs as **linux/amd64**: every metric, the selected models, the exported network, and the
holdout predictions match, with floating-point noise below 1e-15 in pyAgrum's inference outputs
between runs. `tests/compare_results.py` checks a regenerated `app/results.json` against the
committed one at a relative tolerance of 1e-9, and the `reproduce` job in
`.github/workflows/ci.yml` performs that check on every push.

The same locked dependencies on **arm64** (Apple silicon natively, or an arm64 container) do
**not** reproduce the network results. Logistic regression and the random forest match exactly,
but pyAgrum's greedy hill climbing breaks near-ties differently: `BN · AIC` scores 0.259 instead of
0.267 mean validation macro-F1, `BN · BIC` 0.247 instead of 0.256, and the selected network flips
to `BN · BIC + original priors` (0.264). An earlier native run on this Mac showed the same flip and
was first attributed to Python patch versions; the architecture is the actual cause, which is why
`compose.yaml` pins `platform: linux/amd64` (Docker Desktop on Apple silicon emulates it).

Consequences:

- The reference environment for every published number is the pinned image
  `ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim` on amd64 (Python 3.12.12).
- Regenerate artifacts only with `docker compose run --rm experiment`, then run the tests.
  A native `uv run scripts/train.py` on Apple silicon produces a different, internally consistent
  experiment; do not commit its outputs without updating the README and guides.
- The near-tie itself is real: the four network recipes span 0.256–0.267 validation macro-F1
  while the fold-to-fold spread within one recipe is 0.013–0.034. The dashboard's arrow-stability
  view quantifies the same fragility from the bootstrap side.

## Migration notes

- `scripts/train.py` and `tests/test_model.py` still call `sys.path.insert(0, str(ROOT / "src"))`. This is
  now redundant under uv but harmless, and it keeps the no-install path working.
- `scripts/package_download.py` includes `pyproject.toml`, `uv.lock`, and `.python-version` in the archive so
  a downloaded copy can be rebuilt with `uv sync`.
- `.venv/` is already ignored by `.gitignore`.

"""Measure how stable the selected network's arrows are under bootstrap resampling.

Usage: python bootstrap_structure.py [--repeats N]
Relearns the selected recipe on resamples of the recorded training partition and
adds ``network.arc_strength`` to app/results.json. Nothing else in that file,
the selected model, or the holdout evaluation changes.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyagrum as gum

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from wine_quality.model import FEATURES, WineBN, bootstrap_arc_strength

STABLE, WEAK = 0.85, 0.5  # 0.85 is the threshold the original R project used to keep an arc.


def arc_names(network):
    return {(network.variable(a).name(), network.variable(b).name()) for a, b in network.arcs()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=1000,
                        help="number of bootstrap resamples (default 1000)")
    args = parser.parse_args()
    path = ROOT / "app" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    network, split = results["network"], results["split"]
    frame = pd.read_csv(ROOT / "data" / "winequality-red.csv", sep=";")
    frame.columns = [c.replace(" ", "_") for c in frame.columns]
    train = frame.iloc[split["train_indices"]]
    groups = pd.util.hash_pandas_object(train[FEATURES], index=False).to_numpy()
    committed = {tuple(arc) for arc in network["arcs"]}
    # Reproducibility check only: relearn once on the full training partition.
    gum.setNumberOfThreads(1)
    gum.initRandom(split["seed"])
    relearned = arc_names(WineBN.fit(train, network["score"], network["knowledge"]).network)
    print(f"Full-training relearn reproduces {len(committed & relearned)} of {len(committed)} "
          f"committed arcs in this environment (pyAgrum {gum.__version__}).", flush=True)
    strength = bootstrap_arc_strength(train, groups, network["score"], network["knowledge"],
                                      args.repeats, split["seed"])
    lookup = {(p["a"], p["b"]): p for p in strength["pairs"]}
    of_committed = [lookup.get(tuple(sorted(arc)), {"strength": 0.0})["strength"] for arc in committed]
    print(f"Committed arcs: {sum(s >= STABLE for s in of_committed)} in at least {STABLE:.0%} of graphs, "
          f"{sum(WEAK <= s < STABLE for s in of_committed)} between, "
          f"{sum(s < WEAK for s in of_committed)} in under {WEAK:.0%}; "
          f"{strength['skipped']} resamples skipped.", flush=True)
    absent = [p for p in strength["pairs"] if p["strength"] >= WEAK
              and (p["a"], p["b"]) not in committed and (p["b"], p["a"]) not in committed]
    for p in absent:
        print(f"Often learned but absent from the selected graph: {p['a']} – {p['b']} "
              f"({p['strength']:.0%})", flush=True)
    network["arc_strength"] = strength
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved arc strengths from {args.repeats} resamples to {path.relative_to(ROOT)}.", flush=True)


if __name__ == "__main__":
    main()

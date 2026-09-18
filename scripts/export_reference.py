"""Export the validation-selected prediction model for the browser.

Usage: python scripts/export_reference.py
Refits the model that validation selected overall on the recorded training
partition, checks that it reproduces the recorded holdout metrics, and adds
``reference_model`` to app/results.json. Nothing else in that file changes.
"""
import json
import math
from pathlib import Path

from train import ROOT, export_reference, fit_candidate, load_data, metrics, predict


def main():
    path = ROOT / "app" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    name = results["selection"]["overall"]
    frame = load_data()
    train = frame.iloc[results["split"]["train_indices"]]
    test = frame.iloc[results["split"]["test_indices"]]
    fitted = fit_candidate(name, train)
    refit = metrics(test.quality, predict(fitted, test))
    recorded = next(m for m in results["models"] if m["name"] == name)["test"]
    for key, tolerance in (("accuracy", 1e-12), ("macro_f1", 1e-12), ("log_loss", 1e-8)):
        if not math.isclose(refit[key], recorded[key], abs_tol=tolerance):
            raise SystemExit(f"{name} refit does not reproduce the recorded holdout {key}: "
                             f"{refit[key]:.12f} vs {recorded[key]:.12f}. Nothing written.")
    reference = export_reference(fitted)
    if reference is None:
        raise SystemExit(f"{name} is not a model this exporter supports. Nothing written.")
    results["reference_model"] = reference
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {name} as the prediction reference; holdout accuracy {recorded['accuracy']:.1%} reproduced.",
          flush=True)


if __name__ == "__main__":
    main()

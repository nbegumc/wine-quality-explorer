"""Run the prespecified experiment and export the interactive dashboard data.

Usage: python train.py
No test score is used to tune a model, choose a seed, or select a discretizer.
"""
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import pyagrum as gum
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, mean_absolute_error, cohen_kappa_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from wine_quality.model import FEATURES, CLASSES, WineBN, validate_frame

SEED = 123
CANDIDATES = ["Majority baseline", "Logistic regression", "Random forest",
              "BN · BIC", "BN · AIC", "BN · BIC + original priors",
              "BN · AIC + original priors"]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_data():
    path = ROOT / "data" / "winequality-red.csv"
    frame = pd.read_csv(path, sep=";")
    frame.columns = [c.replace(" ", "_") for c in frame.columns]
    validate_frame(frame)
    return frame


def fit_candidate(name, frame):
    if name.startswith("BN"):
        return WineBN.fit(frame, "aic" if "AIC" in name else "bic", "priors" in name)
    if name == "Majority baseline":
        model = DummyClassifier(strategy="prior")
    elif name == "Logistic regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    else:
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=2)
    return model.fit(frame[FEATURES], frame.quality)


def predict(model, frame):
    if isinstance(model, WineBN):
        return model.predict_proba(frame)
    p = model.predict_proba(frame[FEATURES])
    aligned = np.zeros((len(frame), len(CLASSES)))
    for j, label in enumerate(model.classes_):
        aligned[:, int(label) - 3] = p[:, j]
    return aligned


def metrics(y, probabilities):
    predicted = CLASSES[probabilities.argmax(axis=1)]
    report = classification_report(y, predicted, labels=CLASSES,
                                   zero_division=0, output_dict=True)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=CLASSES,
                                      average="macro", zero_division=0)),
            "macro_recall": float(np.mean([report[str(q)]["recall"] for q in CLASSES])),
            "mae": float(mean_absolute_error(y, predicted)),
            "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
            "quadratic_kappa": float(cohen_kappa_score(y, predicted, labels=CLASSES,
                                                       weights="quadratic")),
            "within_one": float(np.mean(np.abs(np.asarray(y) - predicted) <= 1)),
            "confusion": confusion_matrix(y, predicted, labels=CLASSES).tolist(),
            "per_class": [{"quality": int(q), **report[str(q)]} for q in CLASSES]}


def calibration(y, probabilities):
    confidence = probabilities.max(axis=1)
    correct = CLASSES[probabilities.argmax(axis=1)] == np.asarray(y)
    bins = []
    for lo in np.arange(0, 1, .1):
        hi = lo + .1
        mask = (confidence >= lo) & (confidence < hi if hi < .999 else confidence <= 1)
        if mask.any():
            bins.append({"confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean()), "count": int(mask.sum())})
    return bins


def group_bootstrap_interval(y, probabilities, groups, repeats=500):
    """Percentile interval conditional on the fitted model and chosen holdout."""
    rng = np.random.default_rng(SEED)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    predicted = CLASSES[probabilities.argmax(axis=1)]
    y = np.asarray(y)
    scores = []
    for _ in range(repeats):
        sampled = np.concatenate([indices[g] for g in rng.choice(unique, len(unique))])
        scores.append(accuracy_score(y[sampled], predicted[sampled]))
    return np.quantile(scores, [.025, .975]).tolist()


def main():
    gum.setNumberOfThreads(1)
    gum.initRandom(SEED)
    frame = load_data()
    # Group identical predictor rows, including copies with different labels.
    # This preserves all observations without allowing duplicates across splits.
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(frame[FEATURES], frame.quality, groups))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    train_groups, test_groups = groups[train_idx], groups[test_idx]
    assert not set(train_groups) & set(test_groups)
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
              .split(train[FEATURES], train.quality, train_groups))
    for a, b in cv:
        assert not set(train_groups[a]) & set(train_groups[b])
    print(f"Dataset: {len(frame)} rows, {len(set(groups))} predictor groups; "
          f"training {len(train)}, holdout {len(test)}", flush=True)
    results = []
    for name in CANDIDATES:
        fold_metrics = []
        for a, b in cv:
            fitted = fit_candidate(name, train.iloc[a])
            fold_metrics.append(metrics(train.iloc[b].quality, predict(fitted, train.iloc[b])))
        means = {k: float(np.mean([m[k] for m in fold_metrics]))
                 for k in ["accuracy", "macro_f1", "macro_recall", "mae", "log_loss"]}
        results.append({"name": name, "cv": means,
                        "cv_macro_f1_sd": float(np.std([m["macro_f1"] for m in fold_metrics], ddof=1)),
                        "cv_folds": [{k: m[k] for k in means} for m in fold_metrics]})
        print(f"CV {name}: macro-F1={means['macro_f1']:.3f}", flush=True)
    winner = max(results, key=lambda r: r["cv"]["macro_f1"])["name"]
    bn_winner = max([r for r in results if r["name"].startswith("BN")],
                    key=lambda r: r["cv"]["macro_f1"])["name"]
    print(f"Selection BEFORE holdout evaluation: overall={winner}; explorer BN={bn_winner}", flush=True)
    # Decisions above are now frozen. Evaluate the seven prespecified models.
    exported = None
    fitted_bn = None
    bn_probabilities = None
    for result in results:
        fitted = fit_candidate(result["name"], train)
        probabilities = predict(fitted, test)
        result["test"] = metrics(test.quality, probabilities)
        result["calibration"] = calibration(test.quality, probabilities)
        result["accuracy_interval"] = group_bootstrap_interval(test.quality, probabilities, test_groups)
        result["selected_overall"] = result["name"] == winner
        result["selected_bn"] = result["name"] == bn_winner
        if result["name"] == bn_winner:
            exported, fitted_bn, bn_probabilities = fitted.export(), fitted, probabilities
        print(f"Holdout {result['name']}: accuracy={result['test']['accuracy']:.3f}, "
              f"macro-F1={result['test']['macro_f1']:.3f}", flush=True)
    # The explorer uses the very same TRAINING-ONLY model evaluated above.
    ranges = {name: {"min": float(train[name].min()), "max": float(train[name].max()),
                     "median": float(train[name].median())} for name in FEATURES}
    examples = []
    for q in CLASSES:
        matching = test[test.quality == q]
        if len(matching):
            row = matching.iloc[0]
            examples.append({"id": int(row.name), "quality": int(q),
                             "measurements": {f: float(row[f]) for f in FEATURES}})
    reference_queries = [{"evidence": {}, "probabilities": fitted_bn.query({}).tolist()}]
    for row in examples:
        evidence = row["measurements"]
        for names in [FEATURES, ["alcohol"], ["alcohol", "sulphates", "volatile_acidity"]]:
            partial = {name: evidence[name] for name in names}
            reference_queries.append({"evidence": partial,
                                      "probabilities": fitted_bn.query(partial).tolist()})
    for name in FEATURES:
        for value in fitted_bn.bins.cuts[name]:
            partial = {name: value}
            reference_queries.append({"evidence": partial,
                                      "probabilities": fitted_bn.query(partial).tolist()})
    payload = {"schema_version": 1, "dataset": {"rows": len(frame), "features": len(FEATURES),
        "unique_predictor_groups": len(set(groups)), "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {str(q): int((frame.quality == q).sum()) for q in CLASSES},
        "source": "https://archive.ics.uci.edu/dataset/186/wine+quality",
        "sha256": hashlib.sha256((ROOT / 'data/winequality-red.csv').read_bytes()).hexdigest(),
        "license": "CC BY 4.0", "population": "Portuguese red Vinho Verde wines"},
        "split": {"seed": SEED, "train_rows": len(train), "test_rows": len(test),
                  "train_groups": len(set(train_groups)), "test_groups": len(set(test_groups)),
                  "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
                  "cv_folds": [{"train_indices": train_idx[a].tolist(),
                                "validation_indices": train_idx[b].tolist()} for a, b in cv],
                  "test_class_counts": {str(q): int((test.quality == q).sum()) for q in CLASSES}},
        "selection": {"criterion": "mean five-fold validation macro-F1", "overall": winner, "bn": bn_winner},
        "models": results, "network": exported, "ranges": ranges, "examples": examples,
        "reference_queries": reference_queries,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "pyagrum": gum.__version__}}
    write_json(ROOT / "dist" / "results.json", payload)
    write_json(ROOT / "dist" / "data.json", {"columns": FEATURES + ["quality"],
                "rows": frame.to_numpy().tolist()})
    gum.saveBN(fitted_bn.network, str(ROOT / "data" / "selected-network.bif"))
    predictions = test.copy()
    predictions.insert(0, "source_row", predictions.index)
    predictions["prediction"] = CLASSES[bn_probabilities.argmax(axis=1)]
    for j, q in enumerate(CLASSES):
        predictions[f"p_quality_{q}"] = bn_probabilities[:, j]
    predictions.to_csv(ROOT / "data" / "holdout-predictions.csv", index=False)
    print("Saved fitted network, probabilities, split indices, metrics, and dashboard data.", flush=True)


if __name__ == "__main__":
    main()

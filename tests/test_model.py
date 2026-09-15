"""Scientific regression tests, runnable with the standard-library test runner."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wine_quality.model import (FEATURES, QuantileBins, WineBN, bootstrap_arc_strength,
                                validate_frame)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((ROOT / "app/results.json").read_text())
        cls.frame = pd.read_csv(ROOT / "data/winequality-red.csv", sep=";")
        cls.frame.columns = [c.replace(" ", "_") for c in cls.frame.columns]
        cls.model = WineBN(QuantileBins(cls.results["network"]["cuts"]),
                          gum.loadBN(str(ROOT / "data/selected-network.bif")),
                          cls.results["network"]["score"], cls.results["network"]["knowledge"])

    def test_no_group_overlap_in_holdout_or_folds(self):
        groups = pd.util.hash_pandas_object(self.frame[FEATURES], index=False)
        pairs = [(self.results["split"]["train_indices"], self.results["split"]["test_indices"])]
        pairs += [(f["train_indices"], f["validation_indices"]) for f in self.results["split"]["cv_folds"]]
        for a, b in pairs:
            self.assertFalse(set(groups.iloc[a]) & set(groups.iloc[b]))
        a, b = pairs[0]
        self.assertEqual(set(a) | set(b), set(range(len(self.frame))))

    def test_cutpoints_are_training_only(self):
        train = self.frame.iloc[self.results["split"]["train_indices"]]
        fitted = QuantileBins.fit(train)
        self.assertEqual(fitted.cuts, self.results["network"]["cuts"])
        # Changing held-out inputs must not change the fitted transformation.
        adversarial = self.frame.copy()
        adversarial.loc[self.results["split"]["test_indices"], FEATURES] = 99999
        self.assertEqual(QuantileBins.fit(adversarial.iloc[self.results["split"]["train_indices"]]).cuts,
                         fitted.cuts)

    def test_boundary_and_invalid_evidence(self):
        for name, cuts in self.model.bins.cuts.items():
            self.assertEqual(self.model.bins.evidence({name: cuts[0]})[name], "medium")
            self.assertEqual(self.model.bins.evidence({name: cuts[1]})[name], "high")
        with self.assertRaises(ValueError):
            self.model.query({"quality": 7})
        with self.assertRaises(ValueError):
            self.model.query({"alcohol": float("nan")})

    def test_exported_factors_normalize(self):
        for factor in self.results["network"]["factors"]:
            table = np.asarray(factor["values"]).reshape(factor["sizes"])
            np.testing.assert_allclose(table.sum(axis=0), 1, atol=1e-12)
            self.assertTrue((table > 0).all())

    def test_predictions_match_saved_training_model(self):
        # BIF serialization rounds slightly; compare to the exact JSON export.
        for case in self.results["reference_queries"]:
            np.testing.assert_allclose(self.model.query(case["evidence"]),
                                       case["probabilities"], atol=2e-6)

    def test_selection_uses_cv_and_matrix_has_all_rows(self):
        models = self.results["models"]
        self.assertEqual(max(models, key=lambda m: m["cv"]["macro_f1"])["name"],
                         self.results["selection"]["overall"])
        for model in models:
            matrix = np.asarray(model["test"]["confusion"])
            self.assertEqual(int(matrix.sum()), self.results["split"]["test_rows"])
            self.assertAlmostEqual(float(np.trace(matrix) / matrix.sum()), model["test"]["accuracy"])

    def test_reference_model_reproduces_logistic_regression(self):
        reference = self.results.get("reference_model")
        if reference is None:
            self.skipTest("results.json has no reference_model block; run export_reference.py")
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        train = self.frame.iloc[self.results["split"]["train_indices"]]
        test = self.frame.iloc[self.results["split"]["test_indices"]]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(train[FEATURES], train.quality)
        # The browser's arithmetic: standardize, linear logits, softmax.
        z = (test[reference["features"]].to_numpy(dtype=float) - reference["mean"]) / reference["scale"]
        logits = z @ np.asarray(reference["coef"]).T + reference["intercept"]
        exported = np.exp(logits - logits.max(axis=1, keepdims=True))
        exported /= exported.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(exported, model.predict_proba(test[FEATURES]), atol=1e-9)
        recorded = next(m for m in self.results["models"] if m["name"] == reference["name"])
        self.assertTrue(recorded["selected_overall"])
        predicted = np.asarray(reference["classes"])[exported.argmax(axis=1)]
        self.assertAlmostEqual(float(np.mean(predicted == test.quality.to_numpy())), recorded["test"]["accuracy"])

    def assert_valid_arc_strength(self, strength):
        names = set(FEATURES) | {"quality"}
        pairs = [(p["a"], p["b"]) for p in strength["pairs"]]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertGreaterEqual(strength["skipped"], 0)
        for pair in strength["pairs"]:
            self.assertLess(pair["a"], pair["b"])
            self.assertTrue({pair["a"], pair["b"]} <= names)
            self.assertTrue(0 < pair["strength"] <= 1)
            self.assertTrue(0 <= pair["direction"] <= 1)

    def test_bootstrap_arc_strength_is_valid_and_deterministic(self):
        train = self.frame.iloc[self.results["split"]["train_indices"]]
        groups = pd.util.hash_pandas_object(train[FEATURES], index=False).to_numpy()
        first = bootstrap_arc_strength(train, groups, "aic", False, repeats=3, seed=7)
        self.assertEqual(first["repeats"], 3)
        self.assertEqual(first["recipe"], {"score": "aic", "knowledge": False})
        self.assert_valid_arc_strength(first)
        self.assertEqual(first, bootstrap_arc_strength(train, groups, "aic", False, repeats=3, seed=7))

    def test_exported_arc_strength_matches_selected_recipe(self):
        strength = self.results["network"].get("arc_strength")
        if strength is None:
            self.skipTest("results.json has no arc_strength block; run bootstrap_structure.py")
        self.assertGreater(strength["repeats"], 0)
        self.assertEqual(strength["recipe"], {"score": self.results["network"]["score"],
                                              "knowledge": self.results["network"]["knowledge"]})
        self.assert_valid_arc_strength(strength)


if __name__ == "__main__":
    unittest.main()

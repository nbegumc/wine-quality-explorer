"""Train-only discretization, structure learning, and exact inference.

Numerical cut points are estimated from training predictors only. The quality
schema is prespecified from the original red-wine project, never inferred from
held-out labels. All inference conditions on observed predictors, not the target.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyagrum as gum

FEATURES = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"]
CLASSES = np.arange(3, 9)
STATES = ["low", "medium", "high"]
# Preserved only as an experimental condition. These are assumptions, not causes.
ORIGINAL_PRIORS = [("alcohol", "quality"), ("volatile_acidity", "quality"),
                   ("residual_sugar", "density"), ("alcohol", "residual_sugar"),
                   ("sulphates", "quality"), ("pH", "sulphates"),
                   ("total_sulfur_dioxide", "free_sulfur_dioxide")]


def validate_frame(frame, target=True):
    required = FEATURES + (["quality"] if target else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All measurements must be finite and non-missing.")
    if target and not frame.quality.isin(CLASSES).all():
        raise ValueError("This experiment supports the original quality scores 3–8.")


@dataclass
class QuantileBins:
    cuts: dict

    @classmethod
    def fit(cls, frame):
        validate_frame(frame, target=False)
        cuts = {name: np.quantile(frame[name], [1 / 3, 2 / 3]).tolist()
                for name in FEATURES}
        if any(a >= b for a, b in cuts.values()):
            raise ValueError("A feature cannot be divided into three distinct bins.")
        return cls(cuts)

    def transform(self, frame):
        validate_frame(frame, target=False)
        return pd.DataFrame({name: np.asarray(STATES)[
            np.searchsorted(self.cuts[name], frame[name], side="right")]
            for name in FEATURES}, index=frame.index)

    def evidence(self, measurements):
        unknown = set(measurements) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown predictor names: {sorted(unknown)}")
        result = {}
        for name, value in measurements.items():
            if value is None:
                continue
            if not np.isfinite(float(value)):
                raise ValueError("Evidence must contain finite measurements.")
            result[name] = STATES[np.searchsorted(self.cuts[name], value, side="right")]
        return result


def schema_network():
    bn = gum.BayesNet("wine_quality")
    for name in FEATURES:
        bn.add(gum.LabelizedVariable(name, name, STATES))
    bn.add(gum.LabelizedVariable("quality", "quality", [str(q) for q in CLASSES]))
    return bn


@dataclass
class WineBN:
    bins: QuantileBins
    network: object
    score: str
    knowledge: bool

    @classmethod
    def fit(cls, frame, score="bic", knowledge=False):
        validate_frame(frame)
        if score not in {"bic", "aic"}:
            raise ValueError("score must be 'bic' or 'aic'")
        bins = QuantileBins.fit(frame)
        discrete = bins.transform(frame)
        discrete["quality"] = frame.quality.astype(int).astype(str)
        # Passing a DataFrame directly to pyAgrum can leave temporary CSV files
        # in the working directory. Own this short-lived file explicitly.
        with TemporaryDirectory(prefix=".wine-training-", dir=Path.cwd()) as directory:
            path = Path(directory) / "training.csv"
            discrete.to_csv(path, index=False)
            learner = gum.BNLearner(str(path), schema_network())
            network = cls._learn(learner, score, knowledge)
        return cls(bins, network, score, knowledge)

    @staticmethod
    def _learn(learner, score, knowledge):
        learner.setNumberOfThreads(1)
        learner.useGreedyHillClimbing()
        learner.setMaxIndegree(3)
        if score == "bic":
            learner.useScoreBIC()
        else:
            learner.useScoreAIC()
        if knowledge:
            for parent, child in ORIGINAL_PRIORS:
                learner.addMandatoryArc(parent, child)
        # First learn the graph with its stated score, then apply BDeu smoothing
        # only to parameter estimation (equivalent sample size = 5).
        dag = learner.learnDAG()
        learner.useBDeuPrior(5.0)
        network = learner.learnParameters(dag, False)
        return network

    def query(self, measurements):
        inference = gum.LazyPropagation(self.network)
        evidence = self.bins.evidence(measurements)
        if evidence:
            inference.setEvidence(evidence)
        inference.makeInference()
        return np.asarray(inference.posterior("quality").tolist(), dtype=float)

    def predict_proba(self, frame):
        discrete = self.bins.transform(frame)
        inference = gum.LazyPropagation(self.network)
        cache = {}
        result = []
        for values in discrete.itertuples(index=False, name=None):
            if values not in cache:
                inference.eraseAllEvidence()
                inference.setEvidence(dict(zip(FEATURES, values)))
                inference.makeInference()
                cache[values] = np.asarray(inference.posterior("quality").tolist())
            result.append(cache[values])
        return np.asarray(result)

    def export(self):
        """Export CPTs in explicitly defined C-order for browser inference."""
        names = FEATURES + ["quality"]
        factors = []
        for name in names:
            parents = sorted(self.network.variable(p).name()
                             for p in self.network.parents(name))
            variables = [name] + parents
            sizes = [6 if v == "quality" else 3 for v in variables]
            values = [float(self.network.cpt(name)[dict(zip(variables, assignment))])
                      for assignment in product(*(range(s) for s in sizes))]
            factors.append({"variables": variables, "sizes": sizes, "values": values})
        arcs = sorted((self.network.variable(a).name(), self.network.variable(b).name())
                      for a, b in self.network.arcs())
        return {"features": FEATURES, "classes": CLASSES.tolist(), "states": STATES,
                "cuts": self.bins.cuts, "factors": factors, "arcs": arcs,
                "score": self.score, "knowledge": self.knowledge,
                "forced_arcs": ORIGINAL_PRIORS if self.knowledge else []}


def bootstrap_arc_strength(frame, groups, score="bic", knowledge=False, repeats=1000, seed=123):
    """Relearn the structure on group-bootstrap resamples of a training frame.

    For every pair of variables connected in at least one resample, report how
    often an arc appeared in either direction (``strength``) and how often it
    pointed from the alphabetically first name to the second (``direction``).
    This is a stability diagnostic of the learning recipe; it fits no model
    that is used for prediction.
    """
    validate_frame(frame)
    groups = np.asarray(groups)
    if len(groups) != len(frame):
        raise ValueError("groups must align with the rows of frame.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    gum.setNumberOfThreads(1)
    gum.initRandom(seed)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    counts = {}
    skipped = 0
    for repeat in range(repeats):
        sampled = np.concatenate([indices[g] for g in rng.choice(unique, len(unique))])
        try:
            fitted = WineBN.fit(frame.iloc[sampled], score, knowledge)
        except ValueError:
            # A resample can leave a feature without three distinct bins.
            skipped += 1
            continue
        for a, b in fitted.network.arcs():
            names = (fitted.network.variable(a).name(), fitted.network.variable(b).name())
            key = tuple(sorted(names))
            both, forward = counts.get(key, (0, 0))
            counts[key] = (both + 1, forward + int(names == key))
        if (repeat + 1) % 100 == 0:
            print(f"Bootstrap {repeat + 1}/{repeats}", flush=True)
    completed = repeats - skipped
    if not completed:
        raise ValueError("No bootstrap resample could be fitted.")
    pairs = [{"a": a, "b": b, "strength": both / completed, "direction": forward / both}
             for (a, b), (both, forward) in sorted(counts.items())]
    return {"repeats": repeats, "skipped": skipped, "seed": seed,
            "resampling": "training measurement groups with replacement",
            "recipe": {"score": score, "knowledge": knowledge}, "pairs": pairs}

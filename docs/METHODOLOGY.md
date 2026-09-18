# Methodology notes

Why the experiment is built the way it is. New to probability or Bayesian networks? Start with the [beginner guide](BEGINNER_GUIDE.md); this page is the short version for readers who know the vocabulary.

## 1. Split before anything is learned

Bin boundaries, graph structure, and probability tables are all learned from data. If any of them sees the evaluation rows first, the evaluation stops measuring generalisation. The pipeline therefore reserves the holdout from the raw rows before a single cut point is computed, and relearns every data-dependent step inside each validation fold (`WineBN.fit` and the validation loop in `scripts/train.py`).

Think of the holdout as an exam: choosing which relationships to learn after reading the exam answers gives the model information a new wine will never provide.

## 2. Validate the whole recipe, not a fixed graph

Cross-validating a fixed graph only measures how well its tables fit. The question here is whether the *procedure* — bins, search, smoothing — discovers a useful model from fresh training data. Each candidate is therefore fitted from scratch in every fold with hyperparameters fixed in advance, and the winners are chosen by mean validation macro-F1 before the holdout is touched.

## 3. One evaluation loop for every candidate

All seven candidates go through the same code path and are named once, in `CANDIDATES`. A shared loop ties each score to its configuration and makes it impossible to report one model's numbers under another's label.

## 4. Predict through all relevant evidence

A Bayesian network can be queried with any subset of measurements. Inference conditions on every supplied value and sums over the unknown ones — observed children inform a parent just as parents inform a child. The dashboard's JavaScript performs the same variable elimination over the exported tables, and `tests/test_inference.mjs` holds it to pyAgrum's numbers at 1e-10.

## 5. Accuracy is only one view

Scores 5 and 6 make up 82% of the data, so accuracy stays reasonable while extreme scores are missed entirely. Every candidate is reported with accuracy, per-class recall, macro-F1, mean absolute score error, log loss, and a calibration plot. None is sufficient alone; macro-F1 in particular is unstable for classes with a handful of holdout wines, so class counts are shown beside it.

## 6. Duplicate rows are grouped

The dataset contains 240 repeated measurement rows. A random row split can place identical wines on both sides. All rows are kept, but identical predictor vectors form one group that never crosses a split boundary. This is conservative: it does not prove repeated rows are the same physical wine, and it cannot substitute for the producer or batch identifiers the dataset lacks.

## 7. Arrows are not causes

Structure learning finds statistical dependencies; it cannot determine direction where the data does not, and it cannot detect intervention effects. The seven prespecified arcs (`PRIOR_ARCS`) are tested as a constraint condition and are not promoted to chemical facts; the selected network has none. The arrow-stability view (Chapter 13 of the guide) shows how much of the learned structure survives resampling.

## 8. Simplifications, stated

Training-only quantile bins with three levels are transparent and easy to apply to new values, but they discard information: continuous-input baselines lose about three accuracy points when trained on the same bins. Greedy hill climbing with AIC or BIC, at most three parents per node, and BDeu smoothing (equivalent sample size 5) keep tables small enough to estimate from 1,279 rows. These choices are documented so their cost can be judged, not hidden.

## Suggested learning sequence

1. Open the dashboard with no measurements, then add alcohol alone. Explain why the distribution changes.
2. Compare two held-out wines with different true scores. Notice that a confident-looking probability can still be wrong.
3. Read `QuantileBins` and map a value at a cut point into a category by hand.
4. Trace one iteration of the validation loop: what is fitted, and what is held out?
5. Read the model comparison. Explain why logistic regression is the prediction reference while the dashboard queries the network.
6. Run the tests and inspect the exported split indices and probability tables.

For a future experiment, define the change before evaluating the holdout again: an ordinal model, an averaged network as a candidate, or alternative prespecified arcs. Repeatedly improving a model against the current holdout would turn it into validation data.

## References

- [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [Scikit-learn: avoiding leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [pyAgrum learning](https://pyagrum.readthedocs.io/en/latest/BNLearning.html)
- [bnlearn: bootstrap arc strength and model averaging](https://www.bnlearn.com/documentation/man/arc.strength.html)

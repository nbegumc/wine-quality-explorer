# What changed, and what to understand

New to probability or Bayesian networks? Start with [the complete beginner guide](BEGINNER_GUIDE.md), then return here for the focused R audit.

## 1. Splitting after learning still leaks information

In `original/red_wine_final.R`, discretization starts at line 17 and initial graph learning at line 106. The split is made at line 129. Even though probability tables are fitted on training rows, the graph already knows relationships involving test labels.

Think of the test set as an exam. Choosing which relationships to learn after looking at exam answers gives the model information that a new wine will not provide.

The Python pipeline splits raw observations first. In every validation fold, it estimates fresh bin boundaries, graph structure, and conditional probabilities. See `WineBN.fit` and the validation loop in `scripts/train.py`.

## 2. A fixed graph and a learned graph answer different validation questions

The R code passes a graph object into `bn.cv`. That validates parameter fitting given that graph. It does not evaluate the complete process of discovering a graph from new training data. A truly expert-specified graph can be fixed beforehand; one learned from the full dataset cannot be treated as independent of the test folds.

In this rebuild, each model is fitted independently inside each fold. Hyperparameters are fixed in advance; candidate selection uses training validation only.

## 3. Evaluate the model named in the result

The AIC section at line 229 predicts with `fitted_model_bic`. Copy-and-paste errors can create plausible tables whose labels are false. The Python evaluation uses one shared loop over named model configurations, reducing duplicated evaluation code.

## 4. A Bayesian network predicts through all relevant evidence

The R prediction default uses parents of the target. Yet observed children can provide information about a parent. A graph's direction therefore affects what that prediction routine considers.

The Python version uses exact probabilistic inference. The dashboard's JavaScript evaluates the same conditional probability tables with variable elimination. In both cases, unknown measurements are marginalized out. The browser is verified against Python on 41 query cases.

## 5. Accuracy is only one view

Scores 5 and 6 dominate the dataset. Overall accuracy can remain reasonable even when rare classes are missed.

- Accuracy: fraction of exact matches.
- Recall per class: fraction of actual wines in that class correctly identified.
- Macro-F1: average class F1, giving rare classes equal weight.
- Mean absolute score error: average distance between the predicted and actual score.
- Log loss: penalizes assigning little probability to the true class.
- Calibration plot: compares predicted confidence with observed correctness in groups.

None is sufficient alone. Macro-F1 itself is unstable when a class has only a few observations. The held-out class counts are shown next to the results.

## 6. Duplicate rows affect the evaluation

The source contains 240 repeated rows. A random row split can place identical measurements in training and test data. This project retains those observations but groups them by all 11 predictor values, without using quality labels to define the groups.

This is a conservative grouping policy. It does not establish that repeated rows are the same physical wine, nor does it account for unavailable producer or batch identifiers.

## 7. Arrows are not automatically causal

The original seven forced arrows remain available as an experimental condition. They are not promoted to chemical facts. The selected explorer network has no forced arrows in the recorded run.

An association such as P(quality | high alcohol) differs from asking what would happen if alcohol were experimentally changed. No causal intervention claim is supported here. Likewise, two variables can be connected indirectly even without a direct arrow. Independence claims require checking all relevant graph paths.

## 8. This is a rebuild, not exact R parity

Quantile binning is transparent, easy to fit on training data, and easy to apply consistently to new values. It is not Hartemink's algorithm. Changing the discretizer, graph search implementation, constraints, and split changes the experiment. The new scores must not be described as the old experiment rerun in a different language.

The three-category bins also discard information. Continuous-input baseline models help reveal whether that simplification costs predictive performance. pyAgrum was selected for a maintained Python interface to structure learning, BDeu estimation, and exact inference with a compact runtime.

## Suggested learning sequence

1. Open the dashboard with no measurements, then add alcohol alone. Explain why the probability distribution changes.
2. Compare two held-out wines with different true scores. Notice that a confident-looking probability can still be wrong.
3. Read `QuantileBins` and manually map a value at a cut point into a category.
4. Trace one iteration of the validation loop: what is fitted, and what is held out?
5. Read the AIC/BIC model comparison. Explain why the overall chosen model is logistic regression even though the dashboard centers the Bayesian network.
6. Run the tests and inspect the exported split indices and probability tables.

Bootstrap graph stability is now available as a diagnostic (`scripts/bootstrap_structure.py`; see the network view). For a future experiment, define changes before evaluating a new holdout. Options include Hartemink parity, ordinal models, an averaged network as a candidate model, or justified alternative priors. Repeatedly improving a model against the current test scores would gradually turn this holdout into validation data.

## References

- [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [Scikit-learn: avoiding leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [bnlearn cross-validation](https://www.bnlearn.com/documentation/man/bn.cv.html)
- [bnlearn prediction](https://www.bnlearn.com/documentation/man/predict.and.impute.html)
- [bnlearn discretization](https://www.bnlearn.com/documentation/man/preprocessing.html)
- [pyAgrum learning](https://pyagrum.readthedocs.io/en/latest/BNLearning.html)

# Validation record

The following checks passed for the delivered experiment:

- Six Python tests: disjoint predictor groups in holdout and validation; complete outer partition; cut points computed from training only; no change to them when held-out measurements are altered; boundary and invalid-input behavior; normalized positive probability tables; saved-model prediction parity; validation-based selection; and confusion-matrix totals.
- Browser inference matched Python exact inference on 41 saved queries to absolute tolerance 1e-10. Cases include all unknown measurements, partial evidence, all measurements, and both cut points of every feature.
- JavaScript syntax and local static asset references were checked.

The numerical experiment was run using Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, and pyAgrum 3.1.1. The data checksum and all row indices are saved in `app/results.json`.

Full browser-based visual and end-to-end interaction testing was not performed. The optional WebMCP measurement-setting tool is feature-detected; a supported live WebMCP validation context was not available. The standard visible controls do not depend on WebMCP support.

These checks establish implementation consistency, not external predictive validity. They do not establish causation, fairness across wine populations, or strong rare-class performance.

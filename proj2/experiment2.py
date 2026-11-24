"""2: full supplied datasets; stability and overlaps (lecture 5, pp. 22-23)."""

import numpy as np
from common import OUT, RULES, fit, save, training_sets


def main():
    rows = []
    for name, X, _ in training_sets(full_only=True):
        overlaps = (X @ X.T / X.shape[1])[np.triu_indices(len(X), 1)]
        for rule in RULES:
            net = fit(X, rule)
            margin = np.array([x * net.field(x) for x in X])
            stable, fixed = np.all(margin > 0, axis=1).sum(), np.all(margin >= 0, axis=1).sum()
            stats = (np.mean(np.abs(overlaps)), np.all(X == X[0], axis=0).sum())
            rows.append((name, len(X), X.shape[1], rule, stable, fixed, *stats))
    save(
        OUT / "experiment2" / "datasets.csv",
        ["dataset", "p", "n", "rule", "stable", "fixed", "mean_abs_overlap", "constant_pixels"],
        rows,
    )


if __name__ == "__main__":
    main()

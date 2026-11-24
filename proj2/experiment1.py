"""1: number of strictly stable patterns versus training size and learning rule."""

import numpy as np
from common import OUT, RULES, fit, save, training_sets


def main():
    rows = []
    for name, X, indices in training_sets():
        for rule in RULES:
            print(name, len(X), rule)
            net = fit(X, rule)
            for index, x in zip(indices, X):
                margin = np.min(x * net.field(x))
                rows.append((name, len(X), rule, index, margin > 0, margin >= 0, margin))
    save(
        OUT / "experiment1" / "stability.csv",
        ["dataset", "p", "rule", "pattern", "stable", "fixed", "margin"],
        rows,
    )


if __name__ == "__main__":
    main()

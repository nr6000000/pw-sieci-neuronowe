"""3: the larger patterns are the smaller patterns plus 625 constant pixels."""

import numpy as np
from common import BASE, NOISE, OUT, REPEATS, RULES, corrupt, fit, save

FIELD_CHECK_ATOL = 1e-12


def main():
    out = OUT / "experiment3"
    small = np.loadtxt(BASE / "dane/large-25x25.csv", delimiter=",")
    large = np.loadtxt(BASE / "dane/large-25x50.csv", delimiter=",")
    if not np.array_equal(large, np.concatenate((small, np.ones_like(small)), axis=1)):
        raise ValueError("25x50 patterns must match 25x25 patterns plus 625 constant +1 pixels.")
    rows, fields, summary = [], [], []
    rng = np.random.default_rng()
    for name, X in (("large-25x25.csv", small), ("large-25x50.csv", large)):
        print(f"3: {name}")
        models = {rule: fit(X, rule) for rule in RULES}
        h = np.array([models["hebb"].field(x) for x in X])
        fields.append(h)
        overlap = (X @ X.T / X.shape[1])[np.triu_indices(len(X), 1)]
        fixed, strict = np.all(X * h >= 0, axis=1).sum(), np.all(X * h > 0, axis=1).sum()
        stats = (np.all(X == X[0], axis=0).sum(), fixed, strict, overlap.mean())
        rows.append((name, len(X), X.shape[1], *stats))
        for noise in NOISE:
            cases = [(x, corrupt(x, noise, rng)) for x in X for _ in range(REPEATS)]
            for rule, net in models.items():
                for mode in ("sync", "async"):
                    net.synchronous = mode == "sync"
                    trials = []
                    for original, noisy in cases:
                        pred = net.predict(noisy)
                        success = net.last_status == "fixed_point" and np.array_equal(
                            pred, original
                        )
                        trials.append((success, np.mean(pred != original), net.last_steps))
                    mean = np.mean(trials, axis=0)
                    summary.append((name, rule, mode, noise, len(trials), 100 * mean[0], *mean[1:]))
    # Hebb / n: added constant pixels give h_large = h_small / 2 + sum(patterns) / 2.
    n = small.shape[1]
    error = np.max(np.abs(fields[1][:, :n] - 0.5 * fields[0] - 0.5 * small.sum(axis=0)))
    if not np.isfinite(error) or error >= FIELD_CHECK_ATOL:
        raise ValueError(f"Hebb field identity failed (maximum error: {error:.3g}).")
    save(
        out / "metrics.csv",
        [
            "dataset",
            "p",
            "neurons",
            "constant_pixels",
            "fixed",
            "strict",
            "mean_overlap",
            "shift_error",
        ],
        [(*row, error) for row in rows],
    )
    save(
        out / "summary.csv",
        ["dataset", "rule", "mode", "noise", "trials", "success_pct", "mean_error", "mean_steps"],
        summary,
    )


if __name__ == "__main__":
    main()

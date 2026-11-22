"""5: equally sized similar and diverse training sets, with paired corruption masks."""

from collections import defaultdict
from itertools import product

import numpy as np
from common import OUT, RULES, fit, save


def main():
    out = OUT / "experiment5"
    noise_levels = (0.0, 0.1, 0.2, 0.3)
    scores, metrics, stability = defaultdict(list), [], []
    rng = np.random.default_rng()
    for run in range(5):
        similar = np.repeat(rng.choice([-1.0, 1.0], (3, 625)), 2, axis=0)
        for x in similar:
            x[rng.choice(625, round(0.05 * 625), replace=False)] *= -1
        diverse = rng.choice([-1.0, 1.0], (6, 625))
        masks = [rng.permutation(625) for _ in range(6 * len(noise_levels) * 20)]
        for kind, X in (("similar", similar), ("diverse", diverse)):
            print(f"5: run={run}, {kind}")
            distances = np.mean(X[:, None, :] != X[None, :, :], axis=2)
            pairs = np.zeros((6, 6), dtype=bool)
            pairs[[0, 2, 4], [1, 3, 5]] = True
            other = np.triu(np.ones((6, 6), dtype=bool), 1) & ~pairs
            metrics.append((run, kind, distances[pairs].mean(), distances[other].mean()))
            models = {rule: fit(X, rule) for rule in RULES}
            for rule, net in models.items():
                margin = np.array([x * net.field(x) for x in X])
                counts = (np.all(margin >= 0, axis=1).sum(), np.all(margin > 0, axis=1).sum())
                stability.append((run, kind, rule, *counts))
            cases = product(range(6), noise_levels, range(20))
            for (index, noise, _), mask in zip(cases, masks):
                noisy = X[index].copy()
                noisy[mask[: round(noise * 625)]] *= -1
                for rule, mode in product(RULES, ("sync", "async")):
                    net = models[rule]
                    net.synchronous = mode == "sync"
                    pred = net.predict(noisy)
                    success = net.last_status == "fixed_point" and np.array_equal(pred, X[index])
                    scores[kind, rule, mode, noise].append(
                        (success, np.mean(pred != X[index]), net.last_steps)
                    )
    rows = [(*key, len(values), *np.mean(values, axis=0)) for key, values in sorted(scores.items())]
    rows = [(*row[:5], 100 * row[5], *row[6:]) for row in rows]
    save(
        out / "summary.csv",
        ["kind", "rule", "mode", "noise", "trials", "success_pct", "mean_error", "mean_steps"],
        rows,
    )
    save(out / "metrics.csv", ["run", "kind", "paired_hamming", "other_hamming"], metrics)
    save(out / "stability.csv", ["run", "kind", "rule", "fixed", "strict"], stability)


if __name__ == "__main__":
    main()

"""4: recovery versus corruption, dataset type and training size."""

from collections import defaultdict
from itertools import product

import numpy as np
from common import NOISE, OUT, REPEATS, RULES, corrupt, fit, save, training_sets


def main():
    scores, rng = defaultdict(list), np.random.default_rng()
    for name, X, _ in training_sets():
        print(name, len(X))
        models = {rule: fit(X, rule) for rule in RULES}
        for original, noise, _ in product(X, NOISE, range(REPEATS)):
            noisy = corrupt(original, noise, rng)
            for rule, mode in product(RULES, ("sync", "async")):
                net = models[rule]
                net.synchronous = mode == "sync"
                pred = net.predict(noisy)
                success = net.last_status == "fixed_point" and np.array_equal(pred, original)
                scores[name, len(X), rule, mode, noise].append(
                    (success, np.mean(pred != original), net.last_steps, net.last_status)
                )
    rows = []
    for key, values in sorted(scores.items()):
        successes = sum(v[0] for v in values)
        error, steps = np.mean([v[1:3] for v in values], axis=0)
        counts = [
            sum(v[3] == status for v in values)
            for status in ("fixed_point", "cycle_2", "max_steps")
        ]
        rows.append((*key, len(values), successes, error, steps, *counts))
    columns = [
        "dataset",
        "p",
        "rule",
        "mode",
        "noise",
        "trials",
        "successes",
        "mean_error",
        "mean_steps",
        "fixed_point",
        "cycle_2",
        "max_steps",
    ]
    save(OUT / "experiment4" / "summary.csv", columns, rows)


if __name__ == "__main__":
    main()

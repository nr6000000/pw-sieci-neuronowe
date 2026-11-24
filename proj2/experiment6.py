import numpy as np
from common import BASE, OUT, RULES, fit, save

CATEGORIES = ("learned", "negated", "spurious", "cycle_2", "max_steps")
N_INPUTS = 100


def main():
    rng, rows = np.random.default_rng(), []
    for path in sorted((BASE / "dane").glob("*.csv")):
        X = np.loadtxt(path, delimiter=",", ndmin=2)
        inputs = rng.choice([-1.0, 1.0], size=(N_INPUTS, X.shape[1]))
        for rule in RULES:
            print(f"6: {path.name}, {rule}")
            net = fit(X, rule)
            for mode in ("sync", "async"):
                net.synchronous = mode == "sync"
                counts = dict.fromkeys(CATEGORIES, 0)
                for state in inputs:
                    result = net.predict(state)
                    category = net.last_status
                    if category == "fixed_point":
                        category = "spurious"
                        if np.any(np.all(X == result, axis=1)):
                            category = "learned"
                        elif np.any(np.all(-X == result, axis=1)):
                            category = "negated"
                    counts[category] += 1
                for category, count in counts.items():
                    percent = 100 * count / N_INPUTS
                    rows.append((path.name, rule, mode, category, count, percent))
    save(
        OUT / "experiment6" / "summary.csv",
        ["dataset", "rule", "mode", "category", "count", "percent"],
        rows,
    )


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import numpy as np
from hopfield import HopfieldNet

BASE = Path(__file__).resolve().parent
DATA = BASE / "dane"
OUT = BASE / "results"
RULES = ("hebb", "oja")
NOISE = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
REPEATS = 30


def save(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def fit(X, rule):
    # A smaller Oja step keeps the updates finite for wider bitmaps.
    rate = min(0.001, 0.1 / X.shape[1])
    return HopfieldNet(update_rule=rule, learning_rate=rate).fit(X)


def corrupt(original, noise, rng):
    noisy = original.copy()
    noisy[rng.choice(original.size, round(noise * original.size), replace=False)] *= -1
    return noisy


def training_sets(full_only=False):
    # Prefixes give experiments 1, 2 and 4 the same nested training sets.
    for path in sorted(DATA.glob("*.csv")):
        X = np.loadtxt(path, delimiter=",", ndmin=2)
        for p in sorted({1, max(1, len(X) // 2), len(X)}):
            if not full_only or p == len(X):
                yield path.name, X[:p], np.arange(p)

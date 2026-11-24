from itertools import combinations
from math import comb

import numpy as np
from common import OUT, save


def patterns():
    for plus in combinations(range(3, 25), 11):
        for sign in (-1, 1):
            x = -np.ones(25, dtype=int)
            x[:3], x[list(plus)] = sign, 1
            yield x


if __name__ == "__main__":
    folder = OUT / "experiment7"
    # Hebb: means 1,-1/21,0; W*x=(42*x[:3], x[3:]) since sum(last22)=0 (L5,p13/32).
    W = np.zeros((25, 25), dtype=int)
    W[:3, :3], W[3:, 3:] = 21, -1
    np.fill_diagonal(W, 0)
    samples = np.tile(next(patterns()), (100, 1))
    rng = np.random.default_rng()
    for x in samples:
        rng.shuffle(x[3:])
        x[:3] = rng.choice([-1, 1])
        assert np.array_equal(W @ x, np.r_[42 * x[:3], x[3:]])
    cols = [f"x{i}" for i in range(25)]
    count = 2 * comb(22, 11)
    save(folder / "proposal.csv", ["patterns", "neurons", "maximum_proven"], [(count, 25, False)])
    save(folder / "weights_scaled.csv", cols, W)
    save(folder / "examples.csv", cols, samples[:3])

import numpy as np
from hopfield import HopfieldNet

X = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
noisy = np.array([1, 1, 1, -1])
for rule in ("hebb", "oja"):
    for sync in (True, False):
        net = HopfieldNet(synchronous=sync, update_rule=rule).fit(X)
        result = net.predict(noisy)
        assert result.shape == noisy.shape and np.all(np.isin(result, [-1, 1]))
        if rule == "hebb":
            assert np.array_equal(result, X[0])
print("OK: obie reguły i oba tryby działają, Hebb odzyskuje prosty wzorzec.")

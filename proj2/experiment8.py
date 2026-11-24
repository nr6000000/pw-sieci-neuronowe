import numpy as np
from common import OUT, save
from hopfield import HopfieldNet

if __name__ == "__main__":
    train, probe = np.array([[-1, -1], [1, 1]]), np.array([1, -1])
    rows = []
    for sync in (True, False):
        net = HopfieldNet(sync, "hebb").fit(train)
        mode = "sync" if sync else "async"
        states = net.predict(probe, save_steps=True)
        status = net.last_status
        energies = [net.energy(x) for x in states]
        rows += [(mode, k, *x, e, status) for k, (x, e) in enumerate(zip(states, energies))]
        print("8:", mode, [x.tolist() for x in states], energies, status)
    save(OUT / "experiment8" / "states.csv", ["mode", "step", "x0", "x1", "energy", "status"], rows)

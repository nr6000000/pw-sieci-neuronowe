"""9: cached 200x300 photos, Hebb only; dense Oja would require 28.8 GB."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from common import BASE, NOISE, OUT, REPEATS, corrupt, read, save
from hopfield import HopfieldNet
from numba import njit
from PIL import Image


@njit
def bitmap_sweep(X, state):
    state = state.copy()
    overlaps = X @ state
    for i in np.random.permutation(state.size):
        h = -len(X) * state[i]
        for p in range(len(X)):
            h += X[p, i] * overlaps[p]
        old = state[i]
        if h > 0:
            state[i] = 1
        elif h < 0:
            state[i] = -1
        change = state[i] - old
        if change:
            for p in range(len(X)):
                overlaps[p] += X[p, i] * change
    return state


class BitmapHopfield(HopfieldNet):
    # Same Hebb weights, applied via pattern overlaps (lecture 5, slide 16).
    def __init__(self, X):
        super().__init__()
        self.patterns = X
        self.size = X.shape[1]

    def field(self, state):
        X = self.patterns
        return (X.T @ (X @ state) - len(X) * state) / self.size

    def _async_sweep(self, state):
        return bitmap_sweep(self.patterns, state)


def main():
    out = OUT / "experiment9"
    data = BASE / "dane/large_photos"
    paths = [data / row["local_gray"] for row in read(data / "sources.csv")]
    gray = [np.asarray(Image.open(path).convert("L")) for path in paths]
    assert len(gray) >= 3 and all(image.shape == (200, 300) for image in gray)
    X = np.array([np.where(image > np.median(image), 1.0, -1.0).ravel() for image in gray])
    net, n = BitmapHopfield(X), X.shape[1]
    save(
        out / "training.csv",
        ["rule", "status", "p", "neurons", "stored_bytes", "reason"],
        [
            ("hebb", "completed", len(X), n, X.nbytes, ""),
            ("oja", "skipped", len(X), n, "", f"Dense weights require {n * n * 8} bytes"),
        ],
    )
    rng, scores, examples = np.random.default_rng(), defaultdict(list), {}
    for index, original in enumerate(X):
        for noise in NOISE:
            for trial in range(REPEATS):
                noisy = corrupt(original, noise, rng)
                for mode in ("sync", "async"):
                    net.synchronous = mode == "sync"
                    pred = net.predict(noisy)
                    success = net.last_status == "fixed_point" and np.array_equal(pred, original)
                    scores[mode, noise].append((success, np.mean(pred != original), net.last_steps))
                    if noise == 0.1 and trial == 0 and mode == "async":
                        examples[index] = (noisy.copy(), pred.copy())
    rows = [(*key, len(values), *np.mean(values, axis=0)) for key, values in scores.items()]
    rows = [(*row[:3], 100 * row[3], *row[4:]) for row in rows]
    save(
        out / "summary.csv",
        ["mode", "noise", "trials", "success_pct", "mean_error", "mean_steps"],
        rows,
    )
    fig, axes = plt.subplots(len(X), 4, figsize=(12, 2 * len(X)))
    for i in range(len(X)):
        images = (gray[i], X[i], *examples[i])
        for ax, image, title in zip(
            axes[i], images, (paths[i].stem, "Original", "10% noise", "Hebb/async")
        ):
            ax.imshow(np.asarray(image).reshape(200, 300), cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "photos.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()

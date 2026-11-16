from pathlib import Path

import numpy as np
from hopfield import HopfieldNet
from matplotlib import animation
from matplotlib import pyplot as plt


def main():
    hopfield = HopfieldNet(
        synchronous=False,
        update_rule="oja_subspace",
        max_steps=100,
        learning_rate=0.001,
        epochs=10,
        max_fit_steps=1_000_000,
        fit_tol=1e-6,
        random_state=42,
    )

    width = 14
    height = 9

    pattern_index = 3
    noise_fraction = 0.10
    rng = np.random.default_rng(42)

    data_path = Path(__file__).resolve().parent / "dane" / "animals-14x9.csv"
    X = np.genfromtxt(data_path, delimiter=",")
    hopfield.fit(X)

    original = X[pattern_index].copy()
    plt.imshow(original.reshape(height, width))
    plt.show()

    n_flips = round(noise_fraction * original.size)
    noise_indices = rng.choice(original.size, size=n_flips, replace=False)
    x_noisy = original.copy()
    x_noisy[noise_indices] *= -1

    plt.imshow(x_noisy.reshape(height, width))
    plt.show()

    steps = hopfield.predict(x_noisy, save_steps=True)
    print(f"Status: {hopfield.last_status}, kroki: {hopfield.last_steps}")
    pred = steps[-1]
    print(f"Odzyskano oryginal: {np.array_equal(pred, original)}")

    fig, ax = plt.subplots()
    ims = []
    ax.imshow(steps[0].reshape(height, width))
    for step in steps:
        im = ax.imshow(step.reshape(height, width), animated=True)
        ims.append([im])

    _ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
    plt.show()

    plt.imshow(pred.reshape(height, width))
    plt.show()


if __name__ == "__main__":
    main()

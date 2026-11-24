from argparse import ArgumentParser
from pathlib import Path
from runpy import run_path

import numpy as np
from hopfield import HopfieldNet
from matplotlib import animation
from matplotlib import pyplot as plt


def main():
    width, height = 14, 9
    pattern, noise, rule = 4, 0.1, "hebb"
    X = np.loadtxt(Path(__file__).parent / "dane/animals-14x9.csv", delimiter=",")
    net = HopfieldNet(synchronous=False, update_rule=rule).fit(X)
    original = X[pattern]
    noisy = original.copy()
    indices = np.random.default_rng().choice(
        original.size, round(noise * original.size), replace=False
    )
    noisy[indices] *= -1
    states = net.predict(noisy, save_steps=True)
    print(net.last_status, net.last_steps, "odzyskano:", np.array_equal(states[-1], original))
    fig, axes = plt.subplots(1, 3)
    for ax, x, title in zip(axes[:2], (original, noisy), ("Oryginał", "Uszkodzony")):
        ax.imshow(x.reshape(height, width), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
    axes[2].set_title("Kolejne kroki")
    frames = [
        [axes[2].imshow(x.reshape(height, width), cmap="gray", vmin=-1, vmax=1, animated=True)]
        for x in states
    ]
    for ax in axes:
        ax.axis("off")
    _ani = animation.ArtistAnimation(fig, frames, interval=400, repeat_delay=1000)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Demo Hopfielda albo uruchomienie eksperymentów.")
    choices = [str(n) for n in range(1, 10)]
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("experiment", nargs="?", choices=choices + ["all"])
    selection.add_argument("-e", "--experiment", dest="selected", choices=choices)
    selection.add_argument("--all", dest="selected", action="store_const", const="all")
    args = parser.parse_args()
    experiment = args.selected or args.experiment
    if experiment is None:
        main()
    else:
        folder = Path(__file__).resolve().parent
        numbers = range(1, 10) if experiment == "all" else [experiment]
        for number in numbers:
            run_path(str(folder / f"experiment{number}.py"), run_name="__main__")
        if experiment == "all":
            run_path(str(folder / "plot_results.py"), run_name="__main__")

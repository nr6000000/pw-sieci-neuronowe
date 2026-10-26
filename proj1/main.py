import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from experiments import Experiments, Result

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
DEFAULT_OUT = HERE.parent / "experiment_results"


def load_data(
    name: str,
    count: int,
    split_seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task = "classification" if name == "simple" else "regression"
    path = DATA / task / f"data.{name}.train.{count}.csv"
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    expected_columns = 3 if task == "classification" else 2
    if raw.shape[1] != expected_columns or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid data in {path}")

    X, y = raw[:, :-1], raw[:, -1:]
    if task == "classification":
        y = y - 1  # CSV labels 1, 2 -> sigmoid targets 0, 1.
        if not np.isin(y, (0, 1)).all():
            raise ValueError("The simple dataset must contain binary labels 1 and 2")

    order = np.random.default_rng(split_seed).permutation(len(X))
    n_val = min(len(X) - 1, max(1, round(len(X) * val_fraction)))
    val, train = order[:n_val], order[n_val:]
    X_train, y_train = X[train], y[train]
    X_val, y_val = X[val], y[val]

    if task == "regression":
        # Fit scaling on the training partition only.
        xm, xs = X_train.mean(axis=0), X_train.std(axis=0)
        ym, ys = y_train.mean(axis=0), y_train.std(axis=0)
        xs, ys = np.where(xs == 0, 1.0, xs), np.where(ys == 0, 1.0, ys)
        X_train, X_val = (X_train - xm) / xs, (X_val - xm) / xs
        y_train, y_val = (y_train - ym) / ys, (y_val - ym) / ys

    return X_train, y_train, X_val, y_val


def save_results(results: list[Result], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run": f"run_{i:02d}",
            **r.config,
            "train_metric": float(r.train_metric),
            "validation_metric": float(r.validation_metric),
        }
        for i, r in enumerate(results)
    ]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out / "loss_history.npz",
        **{f"run_{i:02d}": np.asarray(r.loss_history) for i, r in enumerate(results)},
    )


def plot_losses(results: list[Result], dataset: str, out: Path, show: bool) -> None:
    # Non-interactive backend for batch runs / CI.
    if not show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = sorted({(r.config["experiment"], r.config["loss"]) for r in results})
    for kind, loss in groups:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in results:
            if (r.config["experiment"], r.config["loss"]) != (kind, loss):
                continue
            label = f"{r.config['activation']}, hidden={r.config['hidden']}"
            ax.plot(np.arange(1, len(r.loss_history) + 1), r.loss_history, label=label)
        ax.set_title(f"{dataset}: {kind} / {loss}")
        ax.set_xlabel("Training step (batch)")
        ax.set_ylabel("Batch training loss")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{kind}_{loss}_loss.png", dpi=140)
        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLP experiments: CSV + loss histories + PNG")
    parser.add_argument("--dataset", choices=("cube", "activation", "simple"), default="cube")
    parser.add_argument(
        "--only",
        "--experiment",
        dest="only",
        choices=("activation", "depth", "width", "loss", "all"),
        default="activation",
    )
    parser.add_argument(
        "--count",
        type=int,
        choices=(100, 500, 1000, 10000),
        default=100,
        help="Size suffix of the training CSV filename",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--show", action="store_true", help="Also display plot windows")
    parser.add_argument(
        "--out", type=Path, help="Output directory (existing files may be replaced)"
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between 0 and 1")
    if args.seed < 0 or args.split_seed < 0:
        parser.error("seed and split-seed must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    task = "classification" if args.dataset == "simple" else "regression"
    data = load_data(args.dataset, args.count, args.split_seed, args.val_fraction)
    exp = Experiments(
        *data,
        task=task,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    jobs = {
        "activation": exp.varying_activation,
        "depth": exp.varying_hidden_layers,
        "width": exp.varying_neurons,
        "loss": (
            exp.varying_regression_loss if task == "regression" else exp.varying_classification_loss
        ),
    }
    out = args.out or DEFAULT_OUT / args.dataset / args.only / f"seed_{args.seed}"
    print(f"Train: {len(data[0])}, validation: {len(data[2])}", flush=True)
    if task == "regression":
        print("Regression metrics use standardized targets and Experiments._metric().", flush=True)
    else:
        print("Classification metric: accuracy (0 to 1).", flush=True)

    results: list[Result] = []
    for kind, run in jobs.items():
        if args.only not in ("all", kind):
            continue
        print(f"Running {args.dataset}: {kind}...", flush=True)
        for r in run():
            r.config.update(
                dataset=args.dataset,
                experiment=kind,
                task=task,
                count=args.count,
                split_seed=args.split_seed,
                val_fraction=args.val_fraction,
            )
            if not np.isfinite([r.train_metric, r.validation_metric]).all():
                raise FloatingPointError(f"Non-finite result: {r.config}")
            if not r.loss_history or not np.isfinite(r.loss_history).all():
                raise ValueError(f"Loss history is empty or non-finite: {r.config}")
            results.append(r)
            print(
                f"  hidden={str(r.config['hidden']):20} "
                f"{r.config['activation']:7} {r.config['loss']:4} "
                f"train={r.train_metric:.6f} val={r.validation_metric:.6f}",
                flush=True,
            )

    save_results(results, out)
    plot_losses(results, args.dataset, out, show=args.show)
    print(f"Saved results in: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

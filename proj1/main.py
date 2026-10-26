import argparse
import csv
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiments import Experiments, Result
from matplotlib import pyplot as plt
from plotter import Plotter

DATA = Path(__file__).resolve().parent / "dataset"
DATASETS = ("cube", "activation", "simple", "three_gauss")
RunKey = tuple[str, tuple[int, ...], str, str]


@dataclass
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    xm: np.ndarray
    xs: np.ndarray
    ym: np.ndarray
    ys: np.ndarray


def load_data(name: str, count: int, split_seed: int) -> Data:
    classification = name in ("simple", "three_gauss")
    folder = DATA / ("classification" if classification else "regression")
    train = np.loadtxt(folder / f"data.{name}.train.{count}.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(folder / f"data.{name}.test.{count}.csv", delimiter=",", skiprows=1)

    X, y = train[:, :-1], train[:, -1:]
    X_test, y_test = test[:, :-1], test[:, -1:]
    rng = np.random.default_rng(split_seed)

    # Keep the same split for every configuration and initialization.
    groups = (
        [np.flatnonzero(y[:, 0] == label) for label in np.unique(y)]
        if classification else [np.arange(len(X))]
    )
    train_ids, val_ids = [], []
    for group in groups:
        ids = rng.permutation(group)
        n_val = max(1, round(0.2 * len(ids)))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    if classification:
        y, y_test = y.astype(int) - 1, y_test.astype(int) - 1
        if name == "three_gauss":
            y = np.eye(3)[y.ravel()]
            y_test = np.eye(3)[y_test.ravel()]

    xm, xs = np.zeros(X.shape[1]), np.ones(X.shape[1])
    ym, ys = np.zeros(y.shape[1]), np.ones(y.shape[1])
    if not classification:
        # Fit scaling on the training partition, never on validation or test data.
        xm, xs = X[train_ids].mean(axis=0), X[train_ids].std(axis=0)
        ym, ys = y[train_ids].mean(axis=0), y[train_ids].std(axis=0)
        xs, ys = np.where(xs == 0, 1, xs), np.where(ys == 0, 1, ys)

    return Data(
        X_train=(X[train_ids] - xm) / xs,
        y_train=(y[train_ids] - ym) / ys,
        X_val=(X[val_ids] - xm) / xs,
        y_val=(y[val_ids] - ym) / ys,
        X_test=X_test,
        y_test=y_test,
        xm=xm, xs=xs, ym=ym, ys=ys,
    )


def make_experiment(
    data: Data, name: str, seed: int, args: argparse.Namespace, save_history: bool = False
) -> Experiments:
    return Experiments(
        data.X_train, data.y_train, data.X_val, data.y_val,
        task="classification" if name in ("simple", "three_gauss") else "regression",
        seed=seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_history=save_history,
    )


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish_plot(path: Path, show: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


def labels(y: np.ndarray) -> np.ndarray:
    return y.argmax(axis=1) if y.shape[1] > 1 else (y[:, 0] >= 0.5).astype(int)


def plot_result(data: Data, name: str, result: Result, out: Path, show: bool) -> None:
    X_fit = data.X_train * data.xs + data.xm
    y_fit = data.y_train * data.ys + data.ym
    X_val = data.X_val * data.xs + data.xm
    y_val = data.y_val * data.ys + data.ym
    regression = name in ("cube", "activation")

    # These are the full instructor files, as shown in the dataset descriptions.
    for split, X, y in (
        ("train", np.vstack((X_fit, X_val)), np.vstack((y_fit, y_val))),
        ("test", data.X_test, data.y_test),
    ):
        plt.figure(figsize=(6, 4))
        if regression:
            Plotter.plot_regression_results(X, y, name, split, points=True)
        else:
            Plotter.plot_classification_results(X, labels(y), f"{name}: {split}")
        finish_plot(out / f"vis_{name}_{split}.png", show)

    model = result.model
    if name == "cube":
        prediction = model.predict((data.X_test - data.xm) / data.xs) * data.ys + data.ym
        plt.figure(figsize=(7, 4))
        Plotter.plot_regression_results(X_fit, y_fit, name, "train", points=True)
        Plotter.plot_regression_results(X_val, y_val, name, "validation", points=True)
        Plotter.plot_regression_results(data.X_test, data.y_test, name, "test target")
        Plotter.plot_regression_results(data.X_test, prediction, name, "prediction")
        plt.legend()
        finish_plot(out / "cube_pred.png", show)

        plt.figure(figsize=(7, 4))
        Plotter.plot_losses(result.loss_history)
        finish_plot(out / "cube_loss.png", show)

        plt.figure(figsize=(7, 4))
        for layer in range(len(model.weights)):
            Plotter.plot_lineweight(model.weight_history, (layer, 0, 0))
        plt.legend()
        finish_plot(out / "weights.png", show)

    elif name == "three_gauss":
        prediction = model.predict(data.X_test)
        plt.figure(figsize=(6, 4))
        Plotter.plot_classification_results(
            data.X_test, labels(prediction), "three_gauss: test predictions"
        )
        finish_plot(out / "three_gauss_pred.png", show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLP comparisons and plot the results")
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--count", type=int, choices=(100, 500, 1000, 10000), default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seeds", nargs="+", type=int, default=[97995, 26102, 39117])
    parser.add_argument("--split-seed", type=int, default=31013)
    parser.add_argument("--out", type=Path, default=Path("experiment_results/part1"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    if min(args.seeds) < 0 or args.split_seed < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct nonnegative model seeds and a nonnegative split seed")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out")
    return args


def main() -> None:
    args = parse_args()
    images = args.out / "images"
    images.mkdir(parents=True)
    if not args.show:
        plt.switch_backend("Agg")

    print(f"Python {platform.python_version()}, NumPy {np.__version__}", flush=True)
    print(f"Settings: {vars(args)}", flush=True)
    runs, summaries, test_runs, tests = [], [], [], []

    for name in args.dataset:
        data = load_data(name, args.count, args.split_seed)
        regression = name in ("cube", "activation")
        scale = float(data.ys[0] ** 2) if regression else 1.0
        metric = "MSE" if regression else "accuracy"
        grouped: dict[RunKey, list[tuple[float, float]]] = defaultdict(list)
        print(f"{name}: {len(data.y_train)} train, {len(data.y_val)} validation", flush=True)

        for seed in args.seeds:
            exp = make_experiment(data, name, seed, args)
            jobs = {
                "activation": exp.varying_activation,
                "depth": exp.varying_hidden_layers,
                "width": exp.varying_neurons,
                "loss": (
                    exp.varying_regression_loss if regression else exp.varying_classification_loss
                ),
            }
            for kind, run in jobs.items():
                print(f"  {name}, seed={seed}, {kind}", flush=True)
                for result in run():
                    key = (
                        kind, tuple(result.config["hidden"]),
                        result.config["activation"], result.config["loss"],
                    )
                    train = result.train_metric * scale
                    val = result.validation_metric * scale
                    if not np.isfinite([train, val]).all():
                        raise FloatingPointError(f"Non-finite training result: {result.config}")
                    grouped[key].append((train, val))
                    runs.append({
                        "dataset": name, "experiment": kind, **result.config,
                        "count": args.count, "split_seed": args.split_seed, "metric": metric,
                        "train": train, "validation": val,
                    })
                del result

        local = []
        for key, values in grouped.items():
            scores = np.asarray(values)
            mean = scores.mean(axis=0)
            std = scores.std(axis=0, ddof=1) if len(scores) > 1 else np.zeros(2)
            row = {
                "dataset": name, "experiment": key[0], "hidden": list(key[1]),
                "activation": key[2], "loss": key[3], "metric": metric,
                "train_mean": mean[0], "train_std": std[0],
                "val_mean": mean[1], "val_std": std[1],
            }
            summaries.append(row)
            local.append((key, row))

        # Select on validation. Break ties by depth, then total hidden units.
        key, _ = min(local, key=lambda item: (
            item[1]["val_mean"] if regression else -item[1]["val_mean"],
            len(item[0][1]), sum(item[0][1]),
        ))
        _, hidden, activation, loss = key
        test_scores = []
        for seed in args.seeds:
            save_history = seed == args.seeds[0] and name == "cube"
            exp = make_experiment(data, name, seed, args, save_history=save_history)
            result = exp._run(list(hidden), activation, loss)
            pred = result.model.predict((data.X_test - data.xm) / data.xs) * data.ys + data.ym
            if not np.isfinite(pred).all():
                raise FloatingPointError("Non-finite test prediction")

            score = (
                float(np.mean((pred - data.y_test) ** 2)) if regression
                else float(np.mean(labels(pred) == labels(data.y_test)))
            )
            test_scores.append(score)
            test_runs.append({
                "dataset": name, **result.config, "metric": metric,
                "count": args.count, "split_seed": args.split_seed,
                "test_n": len(data.y_test), "test": score,
            })
            if seed == args.seeds[0]:
                plot_result(data, name, result, images, args.show)
            del result

        tests.append({
            "dataset": name, "hidden": list(hidden), "activation": activation,
            "loss": loss, "metric": metric, "seeds": str(args.seeds),
            "test_n": len(data.y_test), "mean": float(np.mean(test_scores)),
            "std": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
        })
        save_csv(runs, args.out / "runs.csv")
        save_csv(summaries, args.out / "validation.csv")
        save_csv(test_runs, args.out / "test_runs.csv")
        save_csv(tests, args.out / "test.csv")
        print(f"Test {name}: {tests[-1]['mean']:.6f} +/- {tests[-1]['std']:.6f}", flush=True)

    print(f"Results and plots: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

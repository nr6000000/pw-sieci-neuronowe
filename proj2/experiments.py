import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from hopfield import HopfieldNet
from numba import njit

BASE = Path(__file__).resolve().parent
DATA = BASE / "dane"
OUT = BASE / "results" / "subspace_comparison"
SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 10
MAX_STEPS = 100
RULES = ("hebb", "oja", "oja_subspace")
MAX_FIT_STEPS = 1_000_000
FIT_TOL = 1e-6
FIT_CHECK_EVERY = 100
NOISE = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
REPEATS = 30


@njit
def seed_async(seed):
    np.random.seed(seed)


def save_csv(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stability, recall, errors, training = [], [], [], []

    paths = sorted(DATA.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV datasets found in {DATA}")

    for path in paths:
        print(f"Dataset: {path.name}", flush=True)
        X = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
        ids = rng.permutation(len(X))
        sizes = sorted({1, max(1, len(X) // 2), len(X)})

        for p in sizes:
            print(f"  Training size: {p}", flush=True)
            selected = ids[:p]
            train = X[selected]
            models = {}

            for rule in RULES:
                print(f"    Fitting: {rule}", flush=True)
                rate = (
                    min(LEARNING_RATE, 1.0 / train.shape[1])
                    if rule == "oja_subspace"
                    else LEARNING_RATE
                )
                net = HopfieldNet(
                    synchronous=True,
                    update_rule=rule,
                    max_steps=MAX_STEPS,
                    learning_rate=rate,
                    epochs=EPOCHS,
                    max_fit_steps=MAX_FIT_STEPS,
                    fit_tol=FIT_TOL,
                    fit_check_every=FIT_CHECK_EVERY,
                    random_state=SEED,
                )
                np.random.seed(SEED)
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        net.fit(train)
                    if not np.isfinite(net.weights).all():
                        raise FloatingPointError("Non-finite weights")
                except (FloatingPointError, ValueError) as exc:
                    errors.append((path.name, p, rule, str(exc)))
                    continue

                info = net.fit_info
                training.append(
                    (
                        path.name,
                        p,
                        rule,
                        info["algorithm_variant"],
                        info["training_units"],
                        info["fit_steps"],
                        info["data_passes"],
                        info["converged"],
                        info["residual"],
                        info["n_components"],
                        info["training_rank"],
                        info["initialization_seed"],
                        info["fit_seconds"],
                        float(net.learning_rate),
                    )
                )
                if rule == "oja_subspace":
                    print(
                        f"      updates={info['fit_steps']}, "
                        f"residual={info['residual']:.3g}, "
                        f"converged={info['converged']}",
                        flush=True,
                    )
                models[rule] = net
                for index, original in zip(selected, train):
                    updated = np.where(net.weights @ original >= 0, 1, -1)
                    stable = bool(np.array_equal(updated, original))
                    stability.append((path.name, p, rule, int(index), stable))

            for index, original in zip(selected, train):
                for noise in NOISE:
                    for trial in range(REPEATS):
                        seed = int(rng.integers(2**31 - 1))
                        case_rng = np.random.default_rng(seed)
                        noisy = original.copy()
                        flips = round(noise * original.size)
                        indices = case_rng.choice(original.size, flips, replace=False)
                        noisy[indices] *= -1

                        for rule, net in models.items():
                            for synchronous in (True, False):
                                net.synchronous = synchronous
                                if not synchronous:
                                    seed_async(seed)
                                pred = net.predict(noisy)
                                success = bool(
                                    net.last_status == "fixed_point"
                                    and np.array_equal(pred, original)
                                )
                                mode = "sync" if synchronous else "async"
                                recall.append(
                                    (
                                        path.name,
                                        p,
                                        rule,
                                        mode,
                                        int(index),
                                        noise,
                                        trial,
                                        seed,
                                        success,
                                        float(np.mean(pred != original)),
                                        net.last_status,
                                        net.last_steps,
                                    )
                                )

    save_csv("stability.csv", ["dataset", "p", "rule", "pattern", "stable"], stability)
    save_csv(
        "recall.csv",
        [
            "dataset",
            "p",
            "rule",
            "mode",
            "pattern",
            "noise",
            "trial",
            "seed",
            "success",
            "pixel_error",
            "status",
            "steps",
        ],
        recall,
    )
    save_csv("fit_errors.csv", ["dataset", "p", "rule", "error"], errors)
    save_csv(
        "training.csv",
        [
            "dataset",
            "p",
            "rule",
            "algorithm_variant",
            "training_units",
            "fit_steps",
            "data_passes",
            "converged",
            "residual",
            "n_components",
            "training_rank",
            "initialization_seed",
            "fit_seconds",
            "learning_rate",
        ],
        training,
    )
    metadata = {
        "rules": list(RULES),
        "oja_subspace_variant": HopfieldNet.OJA_SUBSPACE_VARIANT,
        "subspace_training": {
            "max_fit_steps": MAX_FIT_STEPS,
            "requested_learning_rate": LEARNING_RATE,
            "learning_rate_policy": "min(requested_learning_rate, 1 / neurons)",
            "actual_learning_rate": "recorded per fit in training.csv",
            "fit_tol": FIT_TOL,
            "check_every": FIT_CHECK_EVERY,
            "initialization_seed": SEED,
            "components": "rank of training matrix",
            "centered": False,
            "qr_retraction": True,
            "batch_update": "CB - B @ (B.T @ CB), C = X.T @ X / P",
            "memory": "B @ B.T with zero diagonal",
            "budget_note": "batch updates are separate from clamped Oja epochs",
        },
        "oja_variant": HopfieldNet.OJA_VARIANT,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "seed": SEED,
        "noise": list(NOISE),
        "repeats": REPEATS,
        "subset_selection": "one nested random subset per size",
        "case_randomness": "seed column controls noise and Numba async order",
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "rows": {
            "stability": len(stability),
            "recall": len(recall),
            "fit_errors": len(errors),
            "training": len(training),
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (BASE / "hopfield.py", BASE / "experiments.py")
        },
        "dataset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    (OUT / "run_info.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()

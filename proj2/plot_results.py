from common import OUT, RULES, read, save
from matplotlib import pyplot as plt


def main():
    stable, recall = {}, {}
    for row in read(OUT / "experiment1/stability.csv"):
        key = row["dataset"], int(row["p"]), row["rule"]
        stable[key] = stable.get(key, 0) + (row["stable"] == "True")
    for row in read(OUT / "experiment4/summary.csv"):
        key = row["dataset"], int(row["p"]), row["rule"], row["mode"]
        recall.setdefault(key, {})[float(row["noise"])] = (
            100 * int(row["successes"]) / int(row["trials"])
        )
    names = sorted({key[0] for key in stable})
    sizes = {name: sorted({p for d, p, rule in stable if d == name}) for name in names}
    rows = [
        (*key, count, 100 * count / key[1], recall[*key, "sync"][0.1], recall[*key, "async"][0.1])
        for key, count in sorted(stable.items())
    ]
    save(
        OUT / "experiment4/table.csv",
        ["dataset", "p", "rule", "stable", "stable_pct", "recall10_sync_pct", "recall10_async_pct"],
        rows,
    )
    titles = {
        "stability": "1/2: ścisła stabilność",
        "sync": "4: odzyskiwanie synchroniczne",
        "async": "4: odzyskiwanie asynchroniczne",
        "size": "4: liczność zbioru, szum 10%",
    }
    for kind, title in titles.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, name in zip(axes.flat, names):
            for color, rule in enumerate(RULES):
                if kind == "size":
                    for mode, style in (("sync", "o-"), ("async", "x--")):
                        y = [recall[name, p, rule, mode][0.1] for p in sizes[name]]
                        ax.plot(sizes[name], y, style, color=f"C{color}", label=f"{rule}/{mode}")
                elif kind == "stability":
                    ax.plot(
                        sizes[name], [stable[name, p, rule] for p in sizes[name]], "o-", label=rule
                    )
                else:
                    values = sorted(recall[name, max(sizes[name]), rule, kind].items())
                    ax.plot(
                        [100 * f for f, score in values],
                        [score for f, score in values],
                        "o-",
                        label=rule,
                    )
            ax.set(
                title=name.removesuffix(".csv"),
                xlabel="Szum [%]" if kind in ("sync", "async") else "Liczba wzorców",
                ylim=(0, max(sizes[name]) + 1) if kind == "stability" else (-2, 102),
            )
            ax.grid(alpha=0.3)
        fig.suptitle(title + " (Oja: symetryzacja po każdym wzorcu)")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        folder = OUT / ("experiment1" if kind == "stability" else "experiment4")
        fig.savefig(folder / f"{kind}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Summarize genuine execution artifacts and plot the predefined comparisons."""

import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pyamg
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
SEEDS = [20260805, 20260806, 20260807]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "savefig.dpi": 200,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read(method, seed, kind):
    return json.loads((RESULTS / f"{method}_{seed}" / f"{kind}.json").read_text())


def save(fig, name, caption, manifest):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "id": name,
            "path_pdf": f"{name}.pdf",
            "path_png": f"{name}.png",
            "caption": caption,
            "section": "results",
        }
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = []
    for method in ("baseline", "global"):
        for seed in SEEDS:
            t, e = read(method, seed, "training"), read(method, seed, "evaluation")
            metrics = {
                k: v
                for k, v in e["records"]["direct"].items()
                if isinstance(v, (int, float))
            }
            metrics.update(
                seconds=t["seconds"], response_solves=t["summary"]["pre_svd_order"]
            )
            runs.append({"method": method, "seed": seed, "metrics": metrics})
    for seed in SEEDS:
        e = json.loads((RESULTS / f"global_{seed}" / "energy_pod.json").read_text())[
            "K_min"
        ]
        metrics = {k: v for k, v in e.items() if isinstance(v, (int, float))}
        metrics["response_solves"] = read("global", seed, "training")["summary"][
            "pre_svd_order"
        ]
        runs.append({"method": "global_energy_K_min", "seed": seed, "metrics": metrics})
    summary = {}
    for method in ("baseline", "global", "global_energy_K_min"):
        selected = [r["metrics"] for r in runs if r["method"] == method]
        summary[method] = {
            k: {
                "mean": float(np.mean([s[k] for s in selected])),
                "std": float(np.std([s[k] for s in selected], ddof=1)),
                "n_seeds": 3,
            }
            for k in selected[0]
        }
    result = {
        "task": "asymmetric Case-1 parametric thermal reduction",
        "dataset": "MetaHotspot dev@32eb633, 146704 active cells",
        "seeds": SEEDS,
        "metrics": list(summary["baseline"]),
        "summary": summary,
        "runs": runs,
        "provenance": {
            "mode": "measured",
            "source": "results/*/training.json and evaluation.json",
            "dataset_contract": "../data_contract.md",
            "note": (
                "Executed numerical experiment; "
                "no fabricated or physical measurement data"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyamg": pyamg.__version__,
            "hardware": "AMD EPYC 9V74, single-thread sparse and dense solves",
            "base_sha": read("baseline", SEEDS[0], "training")["git_sha"],
        },
        "ablation": {
            m: read(m, 20260805, "evaluation")["records"]
            for m in ("baseline", "global")
        },
    }
    (RESULTS / "results.json").write_text(json.dumps(result, indent=2))
    manifest = []
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("worst_junction_error_K", "response_solves"),
        ("Worst steady junction error [K]", "Full response solves"),
        strict=True,
    ):
        for j, method in enumerate(("baseline", "global")):
            values = [r["metrics"][key] for r in runs if r["method"] == method]
            ax.bar(
                j,
                np.mean(values),
                yerr=np.std(values, ddof=1),
                capsize=4,
                width=0.55,
                color=COLORS[j],
                alpha=0.4,
            )
            ax.scatter(
                j + np.linspace(-0.09, 0.09, 3), values, color=COLORS[j], s=22, zorder=3
            )
        ax.set_xticks([0, 1], ["Original spectrum", "Global interval"])
        ax.set_ylabel(label)
    axes[0].set_title("(a) Held-out accuracy", loc="left")
    axes[1].set_title("(b) Extraction cost", loc="left")
    save(
        fig,
        "fig_01_comparison",
        "Three extraction seeds, individual results and mean +/- sample standard "
        "deviation. Worst junction error over the same 17 physical HTC points; "
        "four 5 W sources.",
        manifest,
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.5), layout="constrained")
    for j, method in enumerate(("baseline", "global")):
        records = read(method, 20260805, "evaluation")["records"]
        keys = ["direct", "svd_0.0001", "svd_1e-05", "svd_1e-06"]
        rr = [records[k] for k in keys]
        ax.semilogy(
            [r["order"] for r in rr],
            [r["worst_junction_error_K"] for r in rr],
            "o-",
            color=COLORS[j],
            label=method,
        )
        path = RESULTS / f"{method}_20260805" / "energy_pod.json"
        if path.exists():
            for name, r in json.loads(path.read_text()).items():
                ax.semilogy(
                    r["order"], r["worst_junction_error_K"], "x", color=COLORS[j], ms=8
                )
                if method == "global":
                    ax.annotate(
                        name,
                        (r["order"], r["worst_junction_error_K"]),
                        xytext={"K_min": (7, -12), "K_nominal": (30, 8), "C": (7, 3)}[
                            name
                        ],
                        textcoords="offset points",
                        fontsize=8,
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": COLORS[j]},
                    )
    ax.set_xlabel("Final state dimension")
    ax.set_ylabel("Worst steady junction error [K]")
    ax.legend(frameon=False)
    ax.set_title(
        "Same snapshots; SVD cutoff 1e-3 to 1e-6\n"
        "Crosses: exploratory fixed-order energy POD"
    )
    save(
        fig,
        "fig_02_compression",
        "First extraction seed only. Lines change normalized snapshot SVD cutoff "
        "using identical responses; crosses change the compression metric at "
        "fixed order. This is an exploratory ablation, not a three-seed comparison.",
        manifest,
    )
    refs = np.load(RESULTS / "reference.npz")
    truth = 308.15 + refs["transient_junction"]
    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 5.0), sharex=True, layout="constrained"
    )
    for i, ax in enumerate(axes.ravel()):
        for j, method in enumerate(("baseline", "global")):
            y = np.asarray(
                read(method, 20260805, "evaluation")["records"]["direct"][
                    "transient_junction_K"
                ]
            )
            ax.plot(
                np.arange(51) * 20,
                (y - truth)[:, i] * 1000,
                color=COLORS[j],
                label=method,
            )
        ax.plot(
            np.arange(51) * 20,
            (308.15 + refs["fl_hist"] - truth)[:, i] * 1000,
            color=COLORS[2],
            ls="--",
            label="FloTHERM ROM",
        )
        energy = json.loads(
            (RESULTS / "global_20260805" / "energy_pod.json").read_text()
        )["K_min"]
        y = np.asarray(energy["transient_junction_K"])
        ax.plot(
            np.arange(51) * 20,
            (y - truth)[:, i] * 1000,
            color=COLORS[3],
            ls=":",
            label="Global + energy POD",
        )
        ax.set_title(f"({chr(97 + i)}) S{i}", loc="left")
        ax.set_ylabel("Junction error [mK]")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    save(
        fig,
        "fig_03_transient",
        "Signed nominal-HTC junction errors relative to the full-order BDF1 "
        "trajectory, first seed, common 20 s time step. Energy POD reuses the "
        "global snapshots at 35 states; FloTHERM denotes the exported 35-state ROM.",
        manifest,
    )
    (FIGURES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

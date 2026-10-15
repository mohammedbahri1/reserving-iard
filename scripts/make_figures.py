"""Génère les figures du README dans docs/figures/. Usage : python scripts/make_figures.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from backtest_large_risks import run as run_backtest  # noqa: E402
from reserving import ChainLadder, MackChainLadder, ODPBootstrap, load_genins  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130, "axes.axisbelow": True,
})
M = 1e6
tri = load_genins()


def development():
    cl = ChainLadder().fit(tri)
    ages = np.arange(1, tri.n + 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i in range(tri.n):
        row = tri.values[i]
        ax.plot(ages[~np.isnan(row)], row[~np.isnan(row)] / M, color=MUTED, linewidth=1, alpha=0.6)
    proj = cl.projected
    for i in range(1, tri.n):
        ax.plot(ages[tri.n - 1 - i:], proj[i, tri.n - 1 - i:] / M, color=BLUE, linewidth=1.2, linestyle="--")
    ax.plot([], [], color=MUTED, linewidth=1, label="Observé")
    ax.plot([], [], color=BLUE, linewidth=1.2, linestyle="--", label="Projeté (Chain Ladder)")
    ax.set_xlabel("Année de développement")
    ax.set_ylabel("Montant cumulé (M€)")
    ax.set_title("Développement des 10 années de survenance", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "developpement.png")
    plt.close(fig)


def mack_by_origin():
    res = MackChainLadder().fit(tri).result()
    x = np.array(tri.origins)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x, res.ibnr / M, color=BLUE, width=0.6, label="Provision Chain Ladder")
    ax.errorbar(x, res.ibnr / M, yerr=1.96 * res.std_error / M, fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=4, label="± 1,96 erreur std de Mack")
    ax.set_xticks(x)
    ax.set_xlabel("Année de survenance")
    ax.set_ylabel("Provision (M€)")
    ax.set_title("Provision par année de survenance et incertitude (Mack)", color=INK)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "mack.png")
    plt.close(fig)


def bootstrap_distribution():
    res = ODPBootstrap(n_sims=20_000).fit(tri).result()
    total = res.total / M
    mean, q995 = total.mean(), np.quantile(total, 0.995)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(total, bins=80, color=BLUE, alpha=0.85)
    for value, label, style in [(mean, f"Moyenne {mean:.1f} M€", "-"), (q995, f"Quantile 99,5 % : {q995:.1f} M€", "--")]:
        ax.axvline(value, color=INK, linewidth=1.3, linestyle=style)
        ax.text(value, ax.get_ylim()[1] * 0.92, "  " + label, color=INK, fontsize=9)
    ax.set_xlabel("Provision totale (M€)")
    ax.set_ylabel("Nombre de simulations")
    ax.set_title("Distribution de la provision totale — bootstrap ODP, 20 000 simulations", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap.png")
    plt.close(fig)


def backtest():
    _, errors = run_backtest()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(-6, 12, 73)
    for (name, e), color in zip(errors.items(), (ORANGE, BLUE)):
        rmse = np.sqrt(np.mean(e ** 2)) / M
        ax.hist(np.clip(e / M, bins[0], bins[-1]), bins=bins, color=color, alpha=0.6,
                label=f"{name} (erreur quadratique {rmse:.1f} M€)")
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Provision estimée − vraie provision (M€), écarts au-delà de 12 M€ regroupés à droite")
    ax.set_ylabel("Nombre de portefeuilles")
    ax.set_title("Grands risques : erreur de provisionnement sur 1 000 portefeuilles simulés", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "backtest_grands_risques.png")
    plt.close(fig)


if __name__ == "__main__":
    development()
    mack_by_origin()
    bootstrap_distribution()
    backtest()
    print(f"Figures écrites dans {OUT}")

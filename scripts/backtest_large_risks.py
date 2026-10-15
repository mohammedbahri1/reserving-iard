"""Backtest : Chain Ladder contre Bornhuetter-Ferguson sur des grands risques simulés.

On simule 1 000 portefeuilles « grands risques » (peu de sinistres, coût de
type Pareto, limite de 5 M€ par sinistre). Pour chacun, on connaît la vraie
provision, ce qui permet de mesurer l'erreur de chaque méthode.

Usage : python scripts/backtest_large_risks.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reserving import BornhuetterFerguson, ChainLadder, incurred_triangle, simulate_large_risks  # noqa: E402

FIRST, LAST, N_PORTFOLIOS = 2006, 2023, 1_000
PREMIUM, FREQUENCY, THRESHOLD, ALPHA, LIMIT = 2.0e6, 0.35, 0.5e6, 1.3, 5.0e6


def expected_loss_ratio(n: int = 2_000_000, seed: int = 0) -> float:
    """S/P théorique du programme : fréquence x coût moyen plafonné / prime."""
    rng = np.random.default_rng(seed)
    capped = np.minimum(THRESHOLD * (1 + rng.pareto(ALPHA, n)), LIMIT)
    return FREQUENCY * capped.mean() / PREMIUM


def run():
    elr = expected_loss_ratio()
    errors = {"Chain Ladder": [], "Bornhuetter-Ferguson": []}
    for seed in range(N_PORTFOLIOS):
        claims, premiums = simulate_large_risks(first_year=FIRST, n_years=LAST - FIRST + 1, seed=seed)
        tri = incurred_triangle(claims, FIRST, LAST)
        ultimate = claims.groupby("id_sinistre")["charge"].max().sum() if len(claims) else 0.0
        true_reserve = ultimate - tri.latest.sum()
        errors["Chain Ladder"].append(ChainLadder().fit(tri).result().total_ibnr - true_reserve)
        errors["Bornhuetter-Ferguson"].append(
            BornhuetterFerguson(premiums.to_numpy(), elr).fit(tri).result().total_ibnr - true_reserve)
    return elr, {k: np.array(v) for k, v in errors.items()}


if __name__ == "__main__":
    elr, errors = run()
    print(f"S/P a priori utilisé par BF : {elr:.1%}\n")
    print(f"{'Méthode':<24}{'Biais moyen':>14}{'Erreur quadratique':>22}{'Intervalle 5-95 %':>28}")
    for name, e in errors.items():
        q5, q95 = np.percentile(e, [5, 95])
        print(f"{name:<24}{e.mean() / 1e3:>11,.0f} k€{np.sqrt(np.mean(e**2)) / 1e3:>19,.0f} k€"
              f"{q5 / 1e3:>13,.0f} à {q95 / 1e3:,.0f} k€".replace(",", " "))

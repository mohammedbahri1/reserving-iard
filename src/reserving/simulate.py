"""Générateur d'un portefeuille automobile fictif, sinistre par sinistre.

Permet de tester les méthodes sur des données « ligne à ligne » réalistes et
de vérifier qu'elles retrouvent la vraie charge ultime, connue par construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["simulate_portfolio"]


def simulate_portfolio(n_claims: int = 10_000, first_year: int = 2015, n_years: int = 10,
                       growth: float = 0.03, mean_cost: float = 2_500.0, cv_cost: float = 1.5,
                       payment_pattern=(0.45, 0.25, 0.12, 0.07, 0.04, 0.03, 0.02, 0.01, 0.007, 0.003),
                       seed: int | None = 7) -> pd.DataFrame:
    """Simule les paiements d'un portefeuille de sinistres.

    Renvoie une ligne par paiement : identifiant, année de survenance,
    année de paiement, montant. Les paiements postérieurs à la dernière
    année sont conservés : ils servent de « vérité » pour juger les méthodes.
    """
    rng = np.random.default_rng(seed)
    pattern = np.asarray(payment_pattern, dtype=float)
    pattern = pattern / pattern.sum()

    # Nombre de sinistres par année, en croissance régulière
    weights = (1 + growth) ** np.arange(n_years)
    years = rng.choice(np.arange(first_year, first_year + n_years), size=n_claims, p=weights / weights.sum())

    # Coût ultime log-normal (moyenne et coefficient de variation donnés)
    sigma2 = np.log(1 + cv_cost ** 2)
    costs = rng.lognormal(np.log(mean_cost) - sigma2 / 2, np.sqrt(sigma2), size=n_claims)

    # Répartition aléatoire du coût dans le temps autour de la cadence moyenne
    shares = rng.dirichlet(pattern * 50, size=n_claims)

    rows = []
    for claim_id, (year, cost, share) in enumerate(zip(years, costs, shares)):
        for dev, s in enumerate(share):
            if s * cost >= 1.0:
                rows.append((claim_id, year, year + dev, s * cost))
    return pd.DataFrame(rows, columns=["id_sinistre", "annee_survenance", "annee_paiement", "montant"])

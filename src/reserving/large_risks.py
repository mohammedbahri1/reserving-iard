"""Provisionnement d'un portefeuille de grands risques (type captive ou réassurance).

Ces portefeuilles ont peu de sinistres, mais très coûteux, et une garantie
plafonnée par sinistre. Leurs triangles sont très volatils : une année peut
passer de zéro au plafond en un an de développement. Le Chain Ladder seul y
devient instable, d'où l'usage :
- de la méthode des sinistres espérés (Expected Loss Ratio) ;
- de Bornhuetter-Ferguson, qui pondère l'observé et l'a priori selon la
  maturité de chaque année ;
- d'un retraitement « as if » : on recalcule l'historique comme si la
  limite actuelle avait toujours été en vigueur, pour rendre les années
  comparables.

Toutes les données de ce module sont simulées.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import ReserveResult
from .triangle import Triangle

__all__ = ["simulate_large_risks", "as_if_limit", "incurred_triangle", "ExpectedLossRatio"]


def simulate_large_risks(first_year: int = 2006, n_years: int = 18, annual_premium: float = 2.0e6,
                         frequency: float = 0.35, pareto_alpha: float = 1.3, threshold: float = 0.5e6,
                         limit: float = 5.0e6, mean_report_delay: float = 1.2, seed: int | None = 5):
    """Simule un programme « grands risques » avec limite par sinistre.

    - nombre de sinistres par an : Poisson(frequency) ;
    - coût : Pareto au-dessus d'un seuil (queue épaisse), plafonné à `limit` ;
    - délai de déclaration géométrique, puis montée en charge de la provision
      dossier jusqu'au coût final.

    Renvoie (sinistres, primes) : une ligne par sinistre et par année de
    développement (charge connue cumulée), et la prime par année.
    """
    rng = np.random.default_rng(seed)
    last_year = first_year + n_years - 1
    rows = []
    claim_id = 0
    for year in range(first_year, last_year + 1):
        for _ in range(rng.poisson(frequency)):
            gross = threshold * (1 + rng.pareto(pareto_alpha))
            delay = rng.geometric(1 / (1 + mean_report_delay)) - 1      # 0, 1, 2... ans
            # Part de la charge finale connue 1, 2, 3 ans après la déclaration
            ramp = np.minimum(1.0, np.array([0.6, 0.85, 1.0, 1.0, 1.0]) * rng.uniform(0.9, 1.1))
            for dev in range(n_years):
                known = 0.0 if dev < delay else gross * ramp[min(dev - delay, len(ramp) - 1)]
                rows.append((claim_id, year, year + dev, known))
            claim_id += 1
    claims = pd.DataFrame(rows, columns=["id_sinistre", "annee_survenance", "annee_vue", "charge_brute"])
    claims = claims.astype({"annee_survenance": int, "annee_vue": int, "charge_brute": float})
    claims["charge"] = np.minimum(claims["charge_brute"], limit)
    premiums = pd.Series(annual_premium, index=range(first_year, last_year + 1), name="prime")
    return claims, premiums


def as_if_limit(claims: pd.DataFrame, limit: float, column: str = "charge_brute") -> pd.DataFrame:
    """Recalcule la charge de chaque sinistre comme si `limit` avait toujours été appliquée."""
    out = claims.copy()
    out["charge"] = np.minimum(out[column], limit)
    return out


def incurred_triangle(claims: pd.DataFrame, first_year: int, valuation_year: int) -> Triangle:
    """Triangle de charges (payés + provisions dossier) vu à une date d'inventaire.

    Les années sans aucun sinistre sont conservées (lignes à zéro) : en grands
    risques, une année blanche est une information.
    """
    df = claims[claims["annee_vue"] <= valuation_year]
    first = first_year
    n = valuation_year - first + 1
    df = df.assign(dev=df["annee_vue"] - df["annee_survenance"])
    pivot = (df.groupby(["annee_survenance", "dev"])["charge"].sum().unstack(fill_value=0.0)
               .reindex(index=range(first, valuation_year + 1), columns=range(n), fill_value=0.0))
    cum = pivot.to_numpy()
    return Triangle(np.where(Triangle.observed_mask(n), cum, np.nan), origins=list(pivot.index))


class ExpectedLossRatio:
    """Méthode des sinistres espérés : ultime = prime x S/P attendu, sans regarder l'observé.

    Par prudence, l'ultime ne descend jamais sous la charge déjà connue.
    """

    def __init__(self, premiums, expected_loss_ratio):
        self.premiums = np.asarray(premiums, dtype=float)
        self.elr = expected_loss_ratio

    def fit(self, triangle: Triangle) -> "ExpectedLossRatio":
        self.triangle = triangle
        return self

    def result(self) -> ReserveResult:
        tri = self.triangle
        ultimate = np.maximum(self.premiums * self.elr, tri.latest)
        return ReserveResult(tri.origins, tri.latest, ultimate)

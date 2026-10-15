"""Méthodes déterministes : Chain Ladder et Bornhuetter-Ferguson."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .triangle import Triangle

__all__ = ["ChainLadder", "BornhuetterFerguson", "ReserveResult"]


@dataclass
class ReserveResult:
    origins: list
    latest: np.ndarray
    ultimate: np.ndarray

    @property
    def ibnr(self) -> np.ndarray:
        """Provision pour sinistres à payer (IBNR au sens large) par année."""
        return self.ultimate - self.latest

    @property
    def total_ibnr(self) -> float:
        return float(self.ibnr.sum())

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame({"Payé à date": self.latest, "Charge ultime": self.ultimate,
                           "Provision": self.ibnr}, index=pd.Index(self.origins, name="Survenance"))
        df.loc["Total"] = df.sum()
        return df


class ChainLadder:
    """Chain Ladder avec facteurs pondérés par les volumes.

    f_k = sum_i C[i, k+1] / sum_i C[i, k], sur les années où les deux sont observés.
    Un facteur de queue (tail) optionnel prolonge le développement au-delà du triangle.
    """

    def __init__(self, tail: float = 1.0):
        if tail < 1.0:
            raise ValueError("Le facteur de queue doit être >= 1")
        self.tail = tail

    def fit(self, triangle: Triangle) -> "ChainLadder":
        self.triangle = triangle
        n, C = triangle.n, triangle.values
        factors = []
        for k in range(n - 1):
            num, den = C[: n - 1 - k, k + 1].sum(), C[: n - 1 - k, k].sum()
            # Colonne entièrement nulle (fréquent en grands risques) : aucun
            # développement observable, on retient un facteur de 1.
            factors.append(num / den if den > 0 else 1.0)
        self.factors_ = np.array(factors)
        # Cumul des facteurs jusqu'à l'ultime (CDF), du développement k à la fin
        self.cdf_ = np.append(np.cumprod(self.factors_[::-1])[::-1], 1.0) * self.tail
        return self

    @property
    def projected(self) -> np.ndarray:
        """Triangle complété (partie inférieure projetée)."""
        C = self.triangle.values.copy()
        n = self.triangle.n
        for i in range(1, n):
            for k in range(n - i, n):
                C[i, k] = C[i, k - 1] * self.factors_[k - 1]
        return C

    def result(self) -> ReserveResult:
        tri = self.triangle
        dev = tri.n - 1 - np.arange(tri.n)          # âge de la diagonale
        ultimate = tri.latest * self.cdf_[dev]
        return ReserveResult(tri.origins, tri.latest, ultimate)

    def development_pattern(self) -> np.ndarray:
        """Cadence de développement : part de la charge ultime connue à chaque âge."""
        return 1.0 / self.cdf_


class BornhuetterFerguson:
    """Bornhuetter-Ferguson : ultime = payé + (1 - 1/CDF) x charge a priori.

    La charge a priori vaut prime x ratio sinistres/primes attendu (ELR).
    La cadence de développement provient du Chain Ladder.
    """

    def __init__(self, premiums, expected_loss_ratio, tail: float = 1.0):
        self.premiums = np.asarray(premiums, dtype=float)
        self.elr = np.broadcast_to(np.asarray(expected_loss_ratio, dtype=float), self.premiums.shape)
        self.tail = tail

    def fit(self, triangle: Triangle) -> "BornhuetterFerguson":
        if len(self.premiums) != triangle.n:
            raise ValueError("Il faut une prime par année de survenance")
        self.triangle = triangle
        self.chain_ladder_ = ChainLadder(self.tail).fit(triangle)
        return self

    def result(self) -> ReserveResult:
        tri = self.triangle
        dev = tri.n - 1 - np.arange(tri.n)
        pct_unreported = 1.0 - 1.0 / self.chain_ladder_.cdf_[dev]
        a_priori = self.premiums * self.elr
        ultimate = tri.latest + pct_unreported * a_priori
        return ReserveResult(tri.origins, tri.latest, ultimate)

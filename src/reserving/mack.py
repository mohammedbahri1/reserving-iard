"""Modèle de Mack (1993) : erreur de prédiction des provisions Chain Ladder.

Mack montre que, sous trois hypothèses (indépendance des années, espérance
conditionnelle E[C_{i,k+1} | C_{i,k}] = f_k C_{i,k}, variance proportionnelle
à C_{i,k}), l'erreur quadratique moyenne de prédiction (MSEP) de la provision
se calcule de façon explicite. Elle combine :
- le risque de processus (aléa de la sinistralité future) ;
- le risque d'estimation (incertitude sur les facteurs f_k).

Référence : T. Mack, « Distribution-free calculation of the standard error of
chain ladder reserve estimates », ASTIN Bulletin 23(2), 1993.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .methods import ChainLadder
from .triangle import Triangle

__all__ = ["MackChainLadder", "MackResult"]


@dataclass
class MackResult:
    origins: list
    latest: np.ndarray
    ultimate: np.ndarray
    std_error: np.ndarray       # erreur standard par année de survenance
    total_std_error: float      # erreur standard de la provision totale

    @property
    def ibnr(self) -> np.ndarray:
        return self.ultimate - self.latest

    @property
    def total_ibnr(self) -> float:
        return float(self.ibnr.sum())

    @property
    def total_cv(self) -> float:
        return self.total_std_error / self.total_ibnr

    def quantile(self, level: float = 0.995) -> float:
        """Quantile de la provision totale sous une hypothèse log-normale."""
        mean, sd = self.total_ibnr, self.total_std_error
        sigma2 = np.log(1.0 + (sd / mean) ** 2)
        mu = np.log(mean) - 0.5 * sigma2
        return float(stats.lognorm.ppf(level, s=np.sqrt(sigma2), scale=np.exp(mu)))

    def summary(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(self.ibnr > 0, self.std_error / self.ibnr, np.nan)
        df = pd.DataFrame({"Payé à date": self.latest, "Charge ultime": self.ultimate,
                           "Provision": self.ibnr, "Erreur std (Mack)": self.std_error,
                           "CV": cv}, index=pd.Index(self.origins, name="Survenance"))
        df.loc["Total"] = [self.latest.sum(), self.ultimate.sum(), self.total_ibnr,
                           self.total_std_error, self.total_cv]
        return df


class MackChainLadder:
    """Chain Ladder stochastique de Mack (sans facteur de queue)."""

    def fit(self, triangle: Triangle) -> "MackChainLadder":
        self.triangle = triangle
        self.cl_ = ChainLadder().fit(triangle)
        n, C, f = triangle.n, triangle.values, self.cl_.factors_

        # Paramètres de variance sigma_k^2
        sigma2 = np.empty(n - 1)
        for k in range(n - 1):
            m = n - 1 - k                                # nombre d'années avec C[., k+1] observé
            if m > 1:
                ratios = C[:m, k + 1] / C[:m, k]
                sigma2[k] = (C[:m, k] * (ratios - f[k]) ** 2).sum() / (m - 1)
        # Dernier sigma : extrapolation proposée par Mack
        if n >= 4:
            sigma2[-1] = min(sigma2[-2] ** 2 / sigma2[-3], sigma2[-3], sigma2[-2])
        else:
            sigma2[-1] = sigma2[-2] if n >= 3 else 0.0
        self.sigma2_ = sigma2

        # Sommes de colonnes utilisées pour estimer chaque facteur
        self.col_sums_ = np.array([C[: n - 1 - k, k].sum() for k in range(n - 1)])
        return self

    def result(self) -> MackResult:
        tri, n = self.triangle, self.triangle.n
        f, s2, S = self.cl_.factors_, self.sigma2_, self.col_sums_
        proj = self.cl_.projected
        ultimate = proj[:, -1]

        mse = np.zeros(n)
        for i in range(1, n):
            ks = np.arange(n - 1 - i, n - 1)
            mse[i] = ultimate[i] ** 2 * np.sum(s2[ks] / f[ks] ** 2 * (1.0 / proj[i, ks] + 1.0 / S[ks]))

        # Erreur de la provision totale : on ajoute la corrélation induite par
        # des facteurs f_k communs à toutes les années.
        total_mse = mse.sum()
        for i in range(1, n):
            ks = np.arange(n - 1 - i, n - 1)
            younger = ultimate[i + 1:].sum()
            total_mse += 2.0 * ultimate[i] * younger * np.sum(s2[ks] / f[ks] ** 2 / S[ks])

        return MackResult(tri.origins, tri.latest, ultimate, np.sqrt(mse), float(np.sqrt(total_mse)))

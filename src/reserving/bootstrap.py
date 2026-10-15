"""Bootstrap du modèle de Poisson surdispersé (ODP), England & Verrall (1999, 2002).

Étapes :
1. ajuster le Chain Ladder et en déduire les incréments « attendus » m_{i,k}
   de la partie observée (en remontant depuis la diagonale) ;
2. calculer les résidus de Pearson r = (X - m) / sqrt(m) et le paramètre
   d'échelle phi, avec la correction de degrés de liberté ;
3. pour chaque simulation : rééchantillonner les résidus, reconstruire un
   pseudo-triangle, réestimer les facteurs (risque d'estimation), projeter
   les incréments futurs et tirer leur valeur dans une loi Gamma de moyenne m
   et de variance phi x m (risque de processus).

On obtient la distribution complète de la provision, donc ses quantiles
(VaR à 99,5 %, comme dans Solvabilité II).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .methods import ChainLadder
from .triangle import Triangle

__all__ = ["ODPBootstrap", "BootstrapResult"]


@dataclass
class BootstrapResult:
    origins: list
    reserves: np.ndarray       # simulations x années de survenance
    scale_phi: float

    @property
    def total(self) -> np.ndarray:
        return self.reserves.sum(axis=1)

    def quantile(self, level: float = 0.995) -> float:
        return float(np.quantile(self.total, level))

    def summary(self, levels=(0.75, 0.95, 0.995)) -> pd.DataFrame:
        cols = {"Moyenne": self.reserves.mean(axis=0), "Erreur std": self.reserves.std(axis=0, ddof=1)}
        for lv in levels:
            cols[f"Q{lv:.1%}".replace(".0%", "%")] = np.quantile(self.reserves, lv, axis=0)
        df = pd.DataFrame(cols, index=pd.Index(self.origins, name="Survenance"))
        tot = self.total
        df.loc["Total"] = [tot.mean(), tot.std(ddof=1)] + [np.quantile(tot, lv) for lv in levels]
        return df


class ODPBootstrap:
    def __init__(self, n_sims: int = 10_000, process: str = "gamma", seed: int | None = 2024):
        if process not in ("gamma", "none"):
            raise ValueError("process doit valoir 'gamma' ou 'none'")
        self.n_sims, self.process, self.seed = n_sims, process, seed

    @staticmethod
    def _fitted_incremental(tri: Triangle, factors: np.ndarray) -> np.ndarray:
        """Incréments attendus de la partie observée, recalculés depuis la diagonale."""
        n = tri.n
        fitted = np.full((n, n), np.nan)
        for i in range(n):
            last = n - 1 - i
            fitted[i, last] = tri.values[i, last]
            for k in range(last - 1, -1, -1):
                fitted[i, k] = fitted[i, k + 1] / factors[k]
        return np.diff(fitted, axis=1, prepend=0.0)

    def fit(self, triangle: Triangle) -> "ODPBootstrap":
        self.triangle = triangle
        n = triangle.n
        mask = Triangle.observed_mask(n)
        self.factors_ = ChainLadder().fit(triangle).factors_
        m = self._fitted_incremental(triangle, self.factors_)
        x = triangle.incremental
        if np.any(m[mask] <= 0):
            raise ValueError("Incréments ajustés négatifs ou nuls : bootstrap ODP non applicable")

        resid = np.where(mask, (x - m) / np.sqrt(np.where(mask, m, 1.0)), np.nan)
        n_obs, n_params = mask.sum(), 2 * n - 1
        self.phi_ = float(np.nansum(resid ** 2) / (n_obs - n_params))
        adj = resid * np.sqrt(n_obs / (n_obs - n_params))
        # Les coins (1re année au dernier âge, dernière année au 1er âge) ont un
        # résidu nul par construction : on les exclut du rééchantillonnage.
        corners = np.zeros_like(mask)
        corners[0, n - 1] = corners[n - 1, 0] = True
        self.residuals_ = adj[mask & ~corners]
        self.fitted_ = m
        return self

    def result(self) -> BootstrapResult:
        rng = np.random.default_rng(self.seed)
        tri, n, B = self.triangle, self.triangle.n, self.n_sims
        mask = Triangle.observed_mask(n)
        m = np.where(mask, self.fitted_, 0.0)

        # 1. Pseudo-triangles incrémentaux (B, n, n)
        draws = rng.choice(self.residuals_, size=(B, n, n), replace=True)
        pseudo = np.where(mask, m + draws * np.sqrt(m), 0.0)
        cum = np.cumsum(pseudo, axis=2)

        # 2. Réestimation des facteurs sur chaque pseudo-triangle (vectorisé)
        factors = np.empty((B, n - 1))
        for k in range(n - 1):
            rows = n - 1 - k
            factors[:, k] = cum[:, :rows, k + 1].sum(axis=1) / cum[:, :rows, k].sum(axis=1)

        # 3. Projection des incréments futurs depuis la diagonale de chaque pseudo-triangle
        #    (England & Verrall : le Chain Ladder est appliqué entièrement aux pseudo-données)
        reserves = np.zeros((B, n))
        for i in range(1, n):
            level = cum[:, i, n - 1 - i].copy()
            for k in range(n - i, n):
                nxt = level * factors[:, k - 1]
                mu = nxt - level
                if self.process == "gamma":
                    shape = np.abs(mu) / self.phi_
                    sim = np.sign(mu) * rng.gamma(np.maximum(shape, 1e-12), self.phi_)
                else:
                    sim = mu
                reserves[:, i] += sim
                level = nxt
        return BootstrapResult(tri.origins, reserves, self.phi_)

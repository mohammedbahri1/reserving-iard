"""Triangle de développement des sinistres.

Convention : `values[i, k]` est le montant cumulé de l'année de survenance i
à l'âge de développement k (0 = première année). Les cellules futures, non
encore observées, valent NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["Triangle"]


class Triangle:
    """Triangle cumulé carré n x n (n années de survenance, n développements)."""

    def __init__(self, cumulative, origins=None):
        values = np.asarray(cumulative, dtype=float)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("Le triangle doit être une matrice carrée n x n")
        n = values.shape[0]
        mask = self.observed_mask(n)
        if np.isnan(values[mask]).any():
            raise ValueError("Valeur manquante dans la partie observée du triangle")
        values = values.copy()
        values[~mask] = np.nan
        self.values = values
        self.origins = list(origins) if origins is not None else list(range(1, n + 1))

    # ------------------------------------------------------------------ #
    @staticmethod
    def observed_mask(n: int) -> np.ndarray:
        """Masque booléen de la partie observée (i + k <= n - 1)."""
        i, k = np.indices((n, n))
        return i + k <= n - 1

    @classmethod
    def from_incremental(cls, incremental, origins=None) -> "Triangle":
        inc = np.asarray(incremental, dtype=float)
        n = inc.shape[0]
        inc = np.where(cls.observed_mask(n), inc, 0.0)
        cum = np.cumsum(inc, axis=1)
        return cls(cum, origins)

    @classmethod
    def from_claims(cls, claims: pd.DataFrame, origin_col: str = "annee_survenance",
                    payment_col: str = "annee_paiement", amount_col: str = "montant") -> "Triangle":
        """Construit un triangle cumulé à partir d'une table de paiements ligne à ligne."""
        df = claims[[origin_col, payment_col, amount_col]].copy()
        first = df[origin_col].min()
        last = df[origin_col].max()
        n = int(last - first + 1)
        df["dev"] = df[payment_col] - df[origin_col]
        df = df[(df["dev"] >= 0) & (df[payment_col] <= last)]
        pivot = (df.groupby([origin_col, "dev"])[amount_col].sum()
                   .unstack(fill_value=0.0)
                   .reindex(index=range(first, last + 1), columns=range(n), fill_value=0.0))
        return cls.from_incremental(pivot.to_numpy(), origins=list(pivot.index))

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def incremental(self) -> np.ndarray:
        inc = np.diff(np.nan_to_num(self.values), axis=1, prepend=0.0)
        inc[~self.observed_mask(self.n)] = np.nan
        return inc

    @property
    def latest(self) -> np.ndarray:
        """Diagonale : dernier montant cumulé connu de chaque année de survenance."""
        return self.values[np.arange(self.n), self.n - 1 - np.arange(self.n)]

    def link_ratios(self) -> np.ndarray:
        """Coefficients de passage individuels C[i, k+1] / C[i, k]."""
        return self.values[:, 1:] / self.values[:, :-1]

    def to_frame(self, incremental: bool = False) -> pd.DataFrame:
        data = self.incremental if incremental else self.values
        return pd.DataFrame(data, index=pd.Index(self.origins, name="Survenance"),
                            columns=pd.Index(range(1, self.n + 1), name="Développement"))

    def __repr__(self) -> str:
        return f"Triangle(n={self.n}, origines={self.origins[0]}..{self.origins[-1]})"

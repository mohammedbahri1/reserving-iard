"""Jeux de données de référence."""
from __future__ import annotations

import numpy as np

from .triangle import Triangle

__all__ = ["load_genins"]

# Triangle de Taylor & Ashe (1983), utilisé par Mack (1993) et England & Verrall (1999).
# Montants incrémentaux, 10 années de survenance x 10 années de développement.
_GENINS_INCREMENTAL = [
    [357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
    [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046],
    [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405],
    [310608, 1108250, 776189, 1562400, 272482, 352053, 206286],
    [443160, 693190, 991983, 769488, 504851, 470639],
    [396132, 937085, 847498, 805037, 705960],
    [440832, 847631, 1131398, 1063269],
    [359480, 1061648, 1443370],
    [376686, 986608],
    [344014],
]


def load_genins() -> Triangle:
    """Triangle « GenIns » de Taylor & Ashe (1983)."""
    n = len(_GENINS_INCREMENTAL)
    inc = np.full((n, n), np.nan)
    for i, row in enumerate(_GENINS_INCREMENTAL):
        inc[i, : len(row)] = row
    return Triangle.from_incremental(inc, origins=list(range(1, n + 1)))

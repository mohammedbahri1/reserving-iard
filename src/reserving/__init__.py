"""reserving : provisionnement IARD.

Triangle            : triangle de développement (cumulé / incrémental)
ChainLadder         : méthode Chain Ladder
BornhuetterFerguson : méthode de Bornhuetter-Ferguson
MackChainLadder     : erreur de prédiction de Mack (1993)
ODPBootstrap        : bootstrap ODP d'England & Verrall, distribution des provisions
simulate_portfolio  : portefeuille automobile fictif ligne à ligne
large_risks         : grands risques (captive), sinistres espérés, retraitement « as if »
"""
from .bootstrap import BootstrapResult, ODPBootstrap
from .datasets import load_genins
from .mack import MackChainLadder, MackResult
from .methods import BornhuetterFerguson, ChainLadder, ReserveResult
from .large_risks import ExpectedLossRatio, as_if_limit, incurred_triangle, simulate_large_risks
from .simulate import simulate_portfolio
from .triangle import Triangle

__all__ = ["Triangle", "ChainLadder", "BornhuetterFerguson", "ReserveResult", "MackChainLadder",
           "MackResult", "ODPBootstrap", "BootstrapResult", "simulate_portfolio", "load_genins",
           "ExpectedLossRatio", "as_if_limit", "incurred_triangle", "simulate_large_risks"]
__version__ = "1.0.0"

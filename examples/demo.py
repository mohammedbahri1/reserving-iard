"""Exemple complet : python examples/demo.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reserving import (BornhuetterFerguson, ChainLadder, MackChainLadder, ODPBootstrap,  # noqa: E402
                       load_genins)

pd.set_option("display.float_format", lambda x: f"{x:,.0f}".replace(",", " "))
pd.set_option("display.width", 120)

tri = load_genins()
print("Triangle cumulé (Taylor & Ashe, 1983)\n")
print(tri.to_frame(), "\n")

cl = ChainLadder().fit(tri)
print("Facteurs de développement :", np.round(cl.factors_, 4), "\n")

mack = MackChainLadder().fit(tri).result()
print("Chain Ladder et erreur de Mack\n")
print(mack.summary().to_string(formatters={"CV": "{:.1%}".format}), "\n")

# Bornhuetter-Ferguson avec des primes illustratives et un S/P a priori de 65 %
premiums = np.linspace(5.0e6, 7.5e6, tri.n)
bf = BornhuetterFerguson(premiums, expected_loss_ratio=0.65).fit(tri).result()
print(f"Provision Bornhuetter-Ferguson (S/P a priori 65 %) : {bf.total_ibnr:,.0f}".replace(",", " "), "\n")

boot = ODPBootstrap(n_sims=20_000).fit(tri).result()
print("Bootstrap ODP (20 000 simulations)\n")
print(boot.summary(), "\n")
fmt = lambda x: f"{x:,.0f}".replace(",", " ")
print("Provision moyenne           :", fmt(boot.total.mean()))
print("Quantile 99,5 %             :", fmt(boot.quantile(0.995)))
print("Marge au-delà de la moyenne :", fmt(boot.quantile(0.995) - boot.total.mean()))

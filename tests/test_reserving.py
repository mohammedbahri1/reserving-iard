"""Tests : valeurs publiées sur le triangle de Taylor & Ashe, cohérence des méthodes."""
import numpy as np
import pytest

from reserving import (BornhuetterFerguson, ChainLadder, MackChainLadder, ODPBootstrap, Triangle,
                       load_genins, simulate_portfolio)


@pytest.fixture(scope="module")
def genins():
    return load_genins()


# --- Triangle --------------------------------------------------------------
def test_incremental_cumulative_round_trip(genins):
    rebuilt = Triangle.from_incremental(genins.incremental)
    np.testing.assert_allclose(rebuilt.values, genins.values, equal_nan=True)


def test_latest_diagonal(genins):
    assert genins.latest[0] == pytest.approx(3_901_463)
    assert genins.latest[-1] == pytest.approx(344_014)


def test_rejects_missing_observed_value():
    bad = np.full((3, 3), np.nan)
    bad[0, 0] = 1.0
    with pytest.raises(ValueError):
        Triangle(bad)


# --- Chain Ladder : valeurs de Mack (1993) ---------------------------------
def test_chain_ladder_factors(genins):
    expected = [3.4906, 1.7473, 1.4574, 1.1739, 1.1038, 1.0863, 1.0539, 1.0766, 1.0177]
    np.testing.assert_allclose(ChainLadder().fit(genins).factors_, expected, atol=1e-4)


def test_chain_ladder_total_reserve(genins):
    assert ChainLadder().fit(genins).result().total_ibnr == pytest.approx(18_680_856, abs=1)


def test_tail_factor_increases_reserve(genins):
    base = ChainLadder().fit(genins).result().total_ibnr
    tailed = ChainLadder(tail=1.05).fit(genins).result().total_ibnr
    assert tailed > base


# --- Bornhuetter-Ferguson --------------------------------------------------
def test_bf_equals_chain_ladder_when_prior_is_cl_ultimate(genins):
    cl = ChainLadder().fit(genins).result()
    bf = BornhuetterFerguson(premiums=cl.ultimate, expected_loss_ratio=1.0).fit(genins).result()
    np.testing.assert_allclose(bf.ultimate, cl.ultimate, rtol=1e-12)


def test_bf_less_sensitive_to_latest_diagonal(genins):
    premiums = np.full(genins.n, 5_000_000.0)
    shocked = Triangle(genins.values * np.where(np.arange(genins.n)[:, None] == genins.n - 1, 1.5, 1.0))
    cl_shift = ChainLadder().fit(shocked).result().ibnr[-1] - ChainLadder().fit(genins).result().ibnr[-1]
    bf = lambda t: BornhuetterFerguson(premiums, 0.65).fit(t).result().ibnr[-1]
    assert abs(bf(shocked) - bf(genins)) < abs(cl_shift)


# --- Mack : erreurs standard publiées --------------------------------------
def test_mack_standard_errors(genins):
    res = MackChainLadder().fit(genins).result()
    expected = [0, 75_535, 121_699, 133_549, 261_406, 411_010, 558_317, 875_328, 971_258, 1_363_155]
    np.testing.assert_allclose(res.std_error, expected, atol=1)
    assert res.total_std_error == pytest.approx(2_447_095, abs=1)


# --- Bootstrap ODP ---------------------------------------------------------
def test_bootstrap_reproducible(genins):
    a = ODPBootstrap(n_sims=500, seed=1).fit(genins).result().total
    b = ODPBootstrap(n_sims=500, seed=1).fit(genins).result().total
    np.testing.assert_array_equal(a, b)


def test_bootstrap_mean_close_to_chain_ladder(genins):
    res = ODPBootstrap(n_sims=20_000, seed=3).fit(genins).result()
    assert res.total.mean() == pytest.approx(18_680_856, rel=0.03)
    # Coefficient de variation proche des ~16 % publiés par England & Verrall
    assert 0.14 < res.total.std() / res.total.mean() < 0.18


def test_scale_parameter(genins):
    assert ODPBootstrap().fit(genins).phi_ == pytest.approx(52_601, rel=1e-3)


# --- Données simulées -------------------------------------------------------
def test_chain_ladder_recovers_true_ultimate():
    claims = simulate_portfolio(n_claims=5_000, seed=11)
    tri = Triangle.from_claims(claims)
    true_ultimate = claims.groupby("annee_survenance")["montant"].sum().sum()
    estimate = ChainLadder().fit(tri).result().ultimate.sum()
    assert estimate == pytest.approx(true_ultimate, rel=0.02)


# --- Grands risques ----------------------------------------------------------
def test_as_if_limit_caps_every_claim():
    from reserving import as_if_limit, simulate_large_risks
    claims, _ = simulate_large_risks(seed=1, limit=10e6)
    capped = as_if_limit(claims, 1e6)
    assert capped["charge"].max() <= 1e6
    assert (capped["charge"] <= claims["charge_brute"]).all()


def test_empty_columns_give_unit_factor():
    tri = Triangle([[0, 0, 0], [0, 0, np.nan], [5, np.nan, np.nan]])
    np.testing.assert_array_equal(ChainLadder().fit(tri).factors_, [1.0, 1.0])


def test_bf_beats_chain_ladder_on_large_risks():
    from reserving import incurred_triangle, simulate_large_risks
    cl_err, bf_err = [], []
    for seed in range(150):
        claims, premiums = simulate_large_risks(seed=seed)
        tri = incurred_triangle(claims, 2006, 2023)
        true = (claims.groupby("id_sinistre")["charge"].max().sum() if len(claims) else 0) - tri.latest.sum()
        cl_err.append(ChainLadder().fit(tri).result().total_ibnr - true)
        bf_err.append(BornhuetterFerguson(premiums.to_numpy(), 0.233).fit(tri).result().total_ibnr - true)
    assert np.sqrt(np.mean(np.square(bf_err))) < np.sqrt(np.mean(np.square(cl_err)))

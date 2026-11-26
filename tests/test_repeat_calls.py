"""Tests for the stored integrals: repeat calls, get_quadrature, the integrand map and the spline."""

import numpy as np
import pylevin
import pytest
from scipy.interpolate import CubicSpline

x = np.geomspace(1e-2, 60, 600)
M = 6


def integrands(x, width):
    return np.stack([x**2 * np.exp(-width * (1 + 0.1 * i) * x**2) * (1 + 0.3 * np.sin(0.2 * x + i)) + 1e-3
                     for i in range(3)], 1)


def integrate(levin, integral_type, result):
    bounds = (np.full(M, x[0]), np.full(M, x[-1]))
    ell = [np.array([0, 1, 3, 6, 10, 15]), np.array([1, 0, 2, 2, 3, 6]), np.array([0, 1, 1, 2, 3, 4])]
    k = [np.geomspace(0.05, 3, M), np.geomspace(0.07, 1.5, M), np.geomspace(0.1, 0.6, M)]
    if integral_type < 2:
        levin.levin_integrate_bessel_single(*bounds, k[0], ell[0], result)
    elif integral_type < 4:
        levin.levin_integrate_bessel_double(*bounds, k[0] / 1.5, k[1], ell[0], ell[1], result)
    else:
        levin.levin_integrate_bessel_triple(*bounds, k[0] / 3, k[1] / 2, k[2], ell[0], ell[1], ell[2], result)


@pytest.mark.parametrize("logy", [False, True])
@pytest.mark.parametrize("integral_type", [0, 1, 2, 4])
def test_quadrature_rule_reproduces_repeat_calls(integral_type, logy):
    """The exported rule does not depend on the integrand: applied to an updated integrand
    (on the same and on a new grid) it reproduces the repeat call."""
    levin = pylevin.pylevin(integral_type, x, integrands(x, 0.01), True, logy, 1)
    levin.set_levin(16, 32, 1e-8, True, False)
    result = np.zeros((M, 3))
    for _ in range(2):
        integrate(levin, integral_type, result)
    stored = levin.get_quadrature()
    for support, width in ((x, 0.015), (np.geomspace(0.9e-2, 61, 700), 0.02)):
        levin.update_integrand(support, integrands(support, width), True, logy)
        integrate(levin, integral_type, result)
        x_nodes, weights, offsets = levin.get_quadrature()
        for before, after in zip(stored, (x_nodes, weights, offsets)):
            np.testing.assert_array_equal(before, after)
        f_nodes = levin.get_integrand(x_nodes)
        from_rule = np.zeros_like(result)
        for p in range(len(offsets) - 1):
            integrand, variable = divmod(p, M)
            nodes = slice(offsets[p], offsets[p + 1])
            from_rule[variable, integrand] = weights[nodes] @ f_nodes[nodes, integrand]
        np.testing.assert_allclose(from_rule, result, rtol=0, atol=1e-13 * np.abs(result).max())


def test_integrand_map_matches_separate_instances():
    """One diagonal instance with an integrand map equals one diagonal instance per multipole."""
    k = np.array([0.1, 0.7, 2.0])
    ells = np.array([0, 4, 12])
    mapped = pylevin.pylevin(0, x, integrands(x, 0.01), True, False, 1, True)
    mapped.set_levin(16, 32, 1e-8, True, False)
    separate = [pylevin.pylevin(0, x, integrands(x, 0.01), True, False, 1, True) for _ in ells]
    for levin in separate:
        levin.set_levin(16, 32, 1e-8, True, False)
    bounds = (np.full(3 * len(ells), x[0]), np.full(3 * len(ells), x[-1]))
    for call in range(3):
        if call == 2:
            for levin in [mapped] + separate:
                levin.update_integrand(x, integrands(x, 0.015), True, False)
        result_mapped = np.zeros(3 * len(ells))
        mapped.levin_integrate_bessel_single(*bounds, np.tile(k, len(ells)), np.repeat(ells, 3), result_mapped,
                                             integrand=np.tile(np.arange(3), len(ells)))
        for i, (levin, ell) in enumerate(zip(separate, ells)):
            result = np.zeros(3)
            levin.levin_integrate_bessel_single(bounds[0][:3], bounds[1][:3], k, np.full(3, ell), result)
            np.testing.assert_array_equal(result_mapped[3 * i:3 * i + 3], result)


@pytest.mark.parametrize("logy", [False, True])
def test_spline_matches_scipy(logy):
    """The integrand is the not-a-knot cubic spline of scipy.interpolate.CubicSpline (in log x)."""
    rng = np.random.default_rng(3)
    for support in (np.geomspace(1e-3, 120, 301), np.sort(np.exp(rng.uniform(np.log(1e-3), np.log(120), 301)))):
        f = integrands(support, 0.01)
        levin = pylevin.pylevin(0, support, f, True, logy, 1)
        x_eval = np.geomspace(support[0], support[-1], 5001)
        expected = CubicSpline(np.log(support), np.log(f) if logy else f)(np.log(x_eval))
        expected = np.exp(expected) if logy else expected
        np.testing.assert_allclose(levin.get_integrand(x_eval), expected, rtol=1e-12, atol=1e-14 * np.abs(expected).max())


def test_invalid_arguments_raise():
    f = integrands(x, 0.01)
    full = pylevin.pylevin(0, x, f, True, False, 1)
    diagonal = pylevin.pylevin(0, x, f, True, False, 1, True)
    bounds = (np.full(3, x[0]), np.full(3, x[-1]))
    k = np.array([0.1, 0.7, 2.0])
    with pytest.raises(ValueError):
        full.levin_integrate_bessel_single(*bounds, k, np.array([0, -1, 2]), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        full.levin_integrate_bessel_single(*bounds, k, np.arange(3), np.zeros((3, 3)), integrand=np.zeros(3, int))
    with pytest.raises(ValueError):
        diagonal.levin_integrate_bessel_single(*bounds, k, np.arange(3), np.zeros(3), integrand=np.full(3, 3))
    with pytest.raises(RuntimeError):
        full.get_quadrature()
    with pytest.raises(ValueError):
        pylevin.pylevin(0, x[::-1], f, True, False, 1)
    with pytest.raises(ValueError):
        pylevin.pylevin(0, x, f[:, 0], True, False, 1)

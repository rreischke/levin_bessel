"""Tests for the ratio cap (max_ratio) and the non-oscillatory quadrature (non_oscillatory)."""

import numpy as np
import pylevin as levin
import pytest


def integrate_single(bessel_type, x, f, k, ell, rel_acc, calls=1, **options):
    lp = levin.pylevin(bessel_type, x, f[:, None], True, True, 1)
    lp.set_levin(8, 32, rel_acc, True, False, 0.0, False, **options)
    result = np.zeros((len(k), 1))
    results = []
    for _ in range(calls):
        lp.levin_integrate_bessel_single(x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k,
                                         np.full(k.shape, ell), result)
        results.append(result[:, 0].copy())
    return results


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_ell_zero_at_small_k(bessel_type):
    """int_0^inf x^(nu+1) e^(-x^2/2) J_nu(kx) dx = k^nu e^(-k^2/2), here for ell = 0.

    For k (b - a) <~ 1 Levin's collocation matrix is nearly singular; without the
    non-oscillatory quadrature the result was off by O(1).
    """
    x = np.geomspace(1e-5, 20, 1000)
    k = np.geomspace(1e-2, 15, 200)
    if bessel_type == 0:
        f, exact = x ** 2 * np.exp(-x ** 2 / 2), np.sqrt(np.pi / 2) * np.exp(-k ** 2 / 2)
    else:
        f, exact = x * np.exp(-x ** 2 / 2), np.exp(-k ** 2 / 2)
    result = integrate_single(bessel_type, x, f, k, 0, 1e-6)[0]
    assert np.max(np.abs(result - exact)) < 1e-6 * np.max(np.abs(exact))


def test_no_false_convergence():
    """Without the ratio cap, ~9% of these integrals reported convergence while off by up to 1e-2."""
    x = np.geomspace(1e-5, 100, 100)
    f = x ** 3 + x ** 2 + x
    k = np.geomspace(0.5, 5, 60)
    lp = levin.pylevin(0, x, f[:, None], True, True, 1)
    lp.set_levin(32, 200, 1e-12, True, False, 0.0, False, max_ratio=1.5)
    truth = np.zeros((len(k), 1))
    lp.levin_integrate_bessel_single(x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k, np.full(k.shape, 5), truth)
    scale = np.abs(truth).max()
    fixed = integrate_single(0, x, f, k, 5, 1e-4)[0]
    old = integrate_single(0, x, f, k, 5, 1e-4, max_ratio=0.0, non_oscillatory=0.0)[0]
    assert np.max(np.abs(fixed - truth[:, 0])) < 1e-4 * scale
    assert np.max(np.abs(old - truth[:, 0])) > 1e-3 * scale


def test_repeat_calls_consistent():
    """Calls 2 and 3 reuse stored subintervals, LU factors and interpolatory weights.

    Previously call 3 differed from call 1 by ~6e-5 here: on the tiny pieces around
    the narrow bump Levin's matrix is nearly singular. Pieces with k dx < 0.5 now
    use interpolatory quadrature; the rest (cond ~1e12) agree to ~1e-9.
    """
    x = np.geomspace(1e-5, 20, 1000)
    f = x ** 7 * np.exp(-x ** 2 / 2) * (1 + 3 * np.exp(-((x - 3) / 0.1) ** 2))
    k = np.geomspace(1e-2, 15, 200)
    first, second, third = integrate_single(0, x, f, k, 5, 1e-4, calls=3)
    scale = np.abs(first).max()
    assert np.max(np.abs(second - first)) < 1e-12 * scale
    assert np.max(np.abs(third - first)) < 1e-7 * scale

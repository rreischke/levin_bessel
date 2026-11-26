import numpy as np
import pytest

import pylevin


@pytest.mark.parametrize("integral_type", [0, 1])
def test_banded_single_bessel_matches_dense(integral_type):
    x = np.linspace(0.2, 18.0, 600)
    integrand = (np.exp(-0.17 * x) * (1.0 + 0.08 * x))[:, None]
    k = np.array([0.4, 2.0, 17.0, 80.0])
    ell = np.array([0, 2, 7, 20], dtype=np.uint32)
    x_min = np.full(k.size, x[0])
    x_max = np.full(k.size, x[-1])

    dense = pylevin.pylevin(integral_type, x, integrand, False, False, 1)
    banded = pylevin.pylevin(integral_type, x, integrand, False, False, 1)
    dense.set_levin(16, 24, 1e-6, True, False, 1e-11, False)
    banded.set_levin(16, 24, 1e-6, True, False, 1e-11, True)

    dense_result = np.zeros((k.size, 1))
    banded_result = np.zeros_like(dense_result)
    dense.levin_integrate_bessel_single(x_min, x_max, k, ell, dense_result)
    banded.levin_integrate_bessel_single(x_min, x_max, k, ell, banded_result)

    np.testing.assert_allclose(banded_result, dense_result, rtol=3e-5, atol=2e-9)


@pytest.mark.parametrize("integral_type", [0, 1])
@pytest.mark.parametrize("n_col", [64, 128])
def test_banded_fft_transform_matches_dense(integral_type, n_col):
    x = np.linspace(0.2, 18.0, 600)
    integrand = (np.exp(-0.17 * x) * (1.0 + 0.08 * x))[:, None]
    k = np.array([0.8, 8.0, 45.0])
    ell = np.array([2, 6, 14], dtype=np.uint32)
    x_min = np.full(k.size, x[0])
    x_max = np.full(k.size, x[-1])

    dense = pylevin.pylevin(integral_type, x, integrand, False, False, 1)
    banded = pylevin.pylevin(integral_type, x, integrand, False, False, 1)
    dense.set_levin(n_col, 16, 1e-6, True, False, 1e-11, False)
    if n_col == 128:
        banded.set_levin(n_col, 16, 1e-6, True, False, 1e-11)
    else:
        banded.set_levin(n_col, 16, 1e-6, True, False, 1e-11, True)

    dense_result = np.zeros((k.size, 1))
    banded_result = np.zeros_like(dense_result)
    dense.levin_integrate_bessel_single(x_min, x_max, k, ell, dense_result)
    banded.levin_integrate_bessel_single(x_min, x_max, k, ell, banded_result)

    np.testing.assert_allclose(banded_result, dense_result, rtol=3e-5, atol=2e-9)


def test_banded_single_falls_back_when_interval_contains_zero():
    x = np.linspace(0.0, 12.0, 500)
    integrand = (1.0 + x)[:, None]
    k = np.array([0.7, 4.0, 30.0])
    ell = np.array([0, 3, 12], dtype=np.uint32)
    x_min = np.zeros(k.size)
    x_max = np.full(k.size, x[-1])

    dense = pylevin.pylevin(0, x, integrand, False, False, 1)
    banded = pylevin.pylevin(0, x, integrand, False, False, 1)
    dense.set_levin(14, 20, 1e-6, True, False, 1e-11, False)
    banded.set_levin(14, 20, 1e-6, True, False, 1e-11, True)

    dense_result = np.zeros((k.size, 1))
    banded_result = np.zeros_like(dense_result)
    dense.levin_integrate_bessel_single(x_min, x_max, k, ell, dense_result)
    banded.levin_integrate_bessel_single(x_min, x_max, k, ell, banded_result)

    np.testing.assert_allclose(banded_result, dense_result, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("integral_type", [0, 1])
def test_banded_single_cache_reuse_after_integrand_updates(integral_type):
    x = np.linspace(0.2, 12.0, 400)
    integrands = [
        (np.exp(-0.2 * x) * (1.0 + 0.05 * x))[:, None],
        (np.exp(-0.1 * x) * (1.0 + 0.02 * x**2))[:, None],
        (np.exp(-0.3 * x) * (1.0 + 0.1 * x))[:, None],
    ]
    k = np.array([0.8, 6.0, 40.0])
    ell = np.array([0, 3, 12], dtype=np.uint32)
    x_min = np.full(k.size, x[0])
    x_max = np.full(k.size, x[-1])

    dense = pylevin.pylevin(integral_type, x, integrands[0], False, False, 1)
    banded = pylevin.pylevin(integral_type, x, integrands[0], False, False, 1)
    dense.set_levin(64, 20, 1e-6, True, False, 1e-11, False)
    banded.set_levin(64, 20, 1e-6, True, False, 1e-11, True)

    for index, integrand in enumerate(integrands):
        if index:
            dense.update_integrand(x, integrand, False, False)
            banded.update_integrand(x, integrand, False, False)
        dense_result = np.zeros((k.size, 1))
        banded_result = np.zeros_like(dense_result)
        dense.levin_integrate_bessel_single(x_min, x_max, k, ell, dense_result)
        banded.levin_integrate_bessel_single(x_min, x_max, k, ell, banded_result)
        np.testing.assert_allclose(banded_result, dense_result, rtol=3e-5, atol=2e-9)

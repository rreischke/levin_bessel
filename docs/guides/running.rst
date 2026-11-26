Running the code
================

This page summarises how to use ``levin``. The `tutorial notebook <https://github.com/rreischke/levin_bessel/blob/main/tutorial/levin_tutorial.ipynb>`_ contains the full examples, with comparisons to other methods and plots. All methods are described in the :doc:`../api`.


A first integral
----------------

``levin`` calculates integrals of the type

.. math::

    I_{ij} = \int_{a_i}^{b_i} f_j(x)\, j_{\ell_i}(k_i x)\, \mathrm{d}x\,,

for ``N`` integrands :math:`f_j(x)` and ``M`` variables :math:`(a_i, b_i, k_i, \ell_i)`, where :math:`j_\ell` is a spherical or cylindrical Bessel function, or a product of two or three of them.

.. code-block:: python

    import numpy as np
    import pylevin as levin

    x = np.geomspace(1e-3, 100, 200)                # support of the integrands
    f = np.stack([x**2*np.exp(-0.05*x**2),
                  x*np.exp(-0.1*x)], axis = 1)      # shape (len(x), N)

    lp = levin.pylevin(0, x, f, True, True, 4)      # type 0: one spherical Bessel function, 4 threads
    lp.set_levin(16, 32, 1e-6, True, False)         # collocation points, maximum bisections, accuracy, boost, verbose

    k = np.geomspace(1e-2, 10, 100)
    ell = np.full(k.size, 2)
    result = np.zeros((k.size, f.shape[1]))         # shape (M, N)
    lp.levin_integrate_bessel_single(np.full(k.size, x[0]), np.full(k.size, x[-1]), k, ell, result)

``result[i, j]`` is the integral over integrand ``j`` for variable ``i``. Products of Bessel functions use the types ``2`` to ``5`` with ``levin_integrate_bessel_double`` and ``levin_integrate_bessel_triple``, which take one ``k`` and ``ell`` array per Bessel function.

The integrands are interpolated with a not-a-knot cubic spline (as ``scipy.interpolate.CubicSpline``), in :math:`\log x` if ``logx = True`` and of :math:`\log f` if ``logy = True``. ``logy`` is applied to each integrand that is positive everywhere; the others are interpolated linearly. Outside of ``x`` the integrands are zero, and ``x`` must be strictly increasing with at least 5 points. ``get_integrand(x)`` returns the interpolated integrands.


Repeated integrals
------------------

An instance remembers the integrals it has calculated, so that calling it again with the same variables is much faster:

1. The first call determines the subintervals (the bisection) for each integral.
2. The second call stores each integral as a quadrature rule, :math:`I = \sum_n w_n f(x_n)`, with weights that do not depend on :math:`f`.
3. Every later call only evaluates these rules for the current integrands.

New integrands are passed with ``update_integrand``. The integration boundaries, ``k`` and ``ell`` must stay the same; otherwise create a new instance. The subintervals were chosen for the first integrands, so new integrands should be similar to them, as is usually the case in inference.

.. code-block:: python

    x_min, x_max = np.full(k.size, x[0]), np.full(k.size, x[-1])
    for _ in range(2):  # bisection, then storing the quadrature rules
        lp.levin_integrate_bessel_single(x_min, x_max, k, ell, result)

    for amplitude in np.linspace(0.5, 1.5, 100):
        lp.update_integrand(x, amplitude*f, True, True)
        lp.levin_integrate_bessel_single(x_min, x_max, k, ell, result)

Repeated calls are fastest with ``logy = False`` and an unchanged ``x``: the integrals are then weighted sums of the new samples, without computing the spline. ``update_integrand`` also accepts a new ``x``, of any length. ``set_levin`` discards the stored integrals.


Diagonal mode and shared integrands
-----------------------------------

With ``diagonal = True`` in the constructor, variable ``i`` only uses integrand ``i`` and ``result`` has shape ``(M)``, with ``M = N``. Often the same integrands are needed for several Bessel orders, for example when :math:`f_j` depends on :math:`k_j` only but the integral is required for many multipoles. The ``integrand`` argument then tells a single instance which integrand each variable uses, so that the integrands are stored and updated only once:

.. code-block:: python

    lp = levin.pylevin(0, x, f_of_k, True, False, 4, True)   # one integrand per k
    lp.set_levin(16, 32, 1e-6, True, False)

    ells = np.arange(2, 100)
    k_all = np.tile(k, len(ells))
    ell_all = np.repeat(ells, len(k))
    integrand_all = np.tile(np.arange(len(k)), len(ells))    # variable i uses integrand integrand_all[i]
    result = np.zeros(len(k_all))
    lp.levin_integrate_bessel_single(np.full(len(k_all), x[0]), np.full(len(k_all), x[-1]),
                                     k_all, ell_all, result, integrand = integrand_all)
    result = result.reshape(len(ells), len(k))

Like ``k`` and ``ell``, the indices must not change between calls.


Exporting the quadrature rules
------------------------------

After the second call, ``get_quadrature()`` returns the stored rules, so that repeated integrals can be evaluated elsewhere, for example with ``jax`` on a GPU, or with the integrands evaluated directly at the nodes instead of interpolated:

.. code-block:: python

    x_nodes, weights, offsets = lp.get_quadrature()
    n_integrals = len(offsets) - 1
    integral = np.repeat(np.arange(n_integrals), np.diff(offsets))   # integral of each node
    f_nodes = my_integrand(x_nodes, integrand_all[integral])         # any function of x and the integrand index
    result = np.bincount(integral, weights*f_nodes, minlength = n_integrals)

The nodes of integral ``p`` are ``x_nodes[offsets[p]:offsets[p + 1]]``; in diagonal mode ``p`` is the variable, otherwise ``result[i, j]`` is ``p = j*M + i``. With ``jax``, ``jax.ops.segment_sum(weights*f_nodes, integral, num_segments = n_integrals)`` does the same and is differentiable.

The rules do not depend on the integrands, so they only have to be exported once: ``update_integrand`` does not change them, and evaluating them for new integrands needs no call to ``levin`` at all.

For integrands given as samples on ``x`` with ``logy = False``, the spline is linear in the samples, so the rules and the spline combine into one matrix ``W``. In the example of the previous section, the integral for multipole ``ells[l]`` over integrand ``j`` is then ``W[j, l] @ f[:, j]``:

.. code-block:: python

    import scipy.sparse as sparse
    from scipy.interpolate import CubicSpline

    rule = sparse.csr_matrix((weights, np.arange(len(x_nodes)), offsets), shape = (n_integrals, len(x_nodes)))
    spline = CubicSpline(np.log(x), np.eye(len(x)))(np.log(x_nodes))   # spline at the nodes for unit samples (logx = True)
    W = np.asarray(rule @ spline).reshape(len(ells), len(k), len(x)).transpose(1, 0, 2).copy()

    f_new = ...                                                          # new samples on x, shape (len(x), len(k))
    result_new = (W @ f_new.T[:, :, None])[:, :, 0].T                    # shape (len(ells), len(k)), no update_integrand needed

For a single integrand set this is about as fast as ``update_integrand`` and a repeated call, since ``levin`` applies the same weights to the samples itself. It pays off for many integrand sets at once, for example in a sampler: with ``f_sets`` of shape ``(len(k), len(x), n_sets)``, ``W @ f_sets`` is one batched matrix product. The nodes must lie inside ``x`` (``levin`` sets the integrands to zero outside, the scipy spline extrapolates). The tutorial shows the full example.


Accuracy and solvers
--------------------

``set_levin(n_col, maximum_bisections, relative_accuracy, super_accurate, verbose, tol_abs = 0.0, banded = True, max_ratio = 5.0, non_oscillatory = 0.5)`` controls the integration:

* ``n_col`` collocation points per subinterval, ``maximum_bisections`` and the relative (``relative_accuracy``) and absolute (``tol_abs``) target accuracy set the adaptive bisection.
* ``super_accurate = True`` uses ``boost`` instead of ``gsl`` for the Bessel functions, which is needed at high orders.
* ``banded`` selects a banded Chebyshev solver for single Bessel functions with at least 64 collocation points. Repeated calls use the stored quadrature rules with either solver.
* ``max_ratio`` and ``non_oscillatory`` guard against wrong results on subintervals spanning many decades in ``x`` and on subintervals where the Bessel functions barely oscillate; ``0`` disables them.

``nthread`` in the constructor sets the number of OpenMP threads, which are used over the integrals and over the integrands in ``update_integrand``.

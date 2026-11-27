This page details the methods and classes provided by the ``levin`` module.

===================
Full Documentation
===================


``levin(type, x, integrand, logx, logy, nthread)``
'''''''''''


Initialises the Levin integrator class given the type of Bessel function 
and if it is a product or not, reads in the integrands and their support
and wether they should be interpolated logarithmically or not.

**Arguments:**

* ``type`` (``integer``): Specifies the type of integral you want to calculate. Currently, the following types are implemented:

    * ``0``: single spherical Bessel function
    
    * ``1``: single cylindrical Bessel function
    
    * ``2``: double spherical Bessel function
    
    * ``3``: double cylindrical Bessel function
    
    * ``4``: triple spherical Bessel function
    
    * ``5``: triple cylindrical Bessel function

* ``x`` (1d ``numpy`` array): One dimensional array over the support of the integrand :math:`x`. The integration used later when integrating should be within the minimum and maximum value of this array.

* ``integrand`` (2d ``numpy`` array): Two dimensional array of the integrands :math:`f(x)`. Needs at least the shape ``len(x), 1``. Generally it should have the shape ``len(x), N``, where ``N`` is the number of different integrals you want to calculate. In other words, you can pass a series of different :math:`f(x)`.
        
* ``logx`` (``True`` / ``False``): Should the integrands be interpolated logarithmically in x or linearly?

* ``logy`` (``True`` / ``False``): Should the integrands, :math:`f(x)`,  be interpolated logarithmically in :math:`y = f(x)` or linearly, automatically checks for each integrand if this is possible.

* ``nthread`` (``integer``): Number of threads used for hyperthreading.

* ``diagonal`` (``True`` / ``False``): *  If ``M = N`` the code can be asked to only calculate the diagonal elements (``True``) of the ``(M,N)`` integrals, or, with the ``integrand`` argument, one given integrand per variable. Default is ``False``. For ``M`` see the description of ``levin_integrate_bessel_single(x_min, x_max, k, ell, result, integrand = None)``.




``set_levin(n_col_in, maximum_number_bisections_in, relative_accuracy_in, super_accurate, verbose, tol_abs_in = 0.0, banded = True, max_ratio = 5.0, non_oscillatory = 0.5)``
'''''''''''

Sets up the internal parameters of the Levin integrator. If not called, default values are used.

**Arguments:**

* ``n_col_in`` (``integer``): Number of collocation points used at which the solution to the differential equation is constructed. **default: 8**

* ``maximum_number_bisections_in`` (``integer``): Maximum number of bisections used for the adaptive integration. **default: 32**

* ``relative_accuracy_in`` (``float``): Relative target accuracy at which the adaptive integration (the bisection) is stopped. A subinterval also counts as unconverged as long as the integrand is not resolved there, i.e. the last Chebyshev coefficients of its interpolant on the collocation points exceed ``relative_accuracy_in`` times the largest one. Like any adaptive error estimate, this can occasionally stop early for a single integral, so choose it somewhat below the accuracy you need. **default: 1e-4**

* ``super_accurate`` (``True`` / ``False``): Should ``gsl`` (``False``)  or ``boost`` (``True``) be used for the computation of the Bessel functions. Especially at higher orders (larger than 100) of the Bessel functions ``gsl``can fail. **default: False**

* ``verbose`` (``True`` / ``False``): Should the code talk to you? **default: False**

* ``tol_abs_in`` (``float``): Absolute target accuracy at which the adaptive integration (the bisection) is stopped. **default: 0.0** 

* ``banded`` (``True`` / ``False``): Use the banded Chebyshev Levin solver for single spherical or cylindrical Bessel integrals (types ``0`` and ``1``). This is enabled by default when at least 64 coefficients are used; smaller systems keep the dense solver. The banded operator is stored and factored using linear memory/work in the number of coefficients. Repeated integrals with the same interval and Bessel parameters reuse the factorization and endpoint corrections; from the third call on, integrals use the stored quadrature weights (see ``get_quadrature``) for either solver. The bounded cache holds at most 128 operator entries and 64 MiB. For more than 32 coefficients, the right-hand side uses a Bluestein FFT for the Clenshaw--Curtis DCT-I. The existing dense solver is used for other integral types and as a fallback for single-Bessel intervals that include or cross ``x = 0`` or where the banded system cannot be factored. **default: True**

* ``max_ratio`` (``float``): Caps :math:`b / \max(a, x_t)` of every subinterval :math:`[a, b]`, where :math:`x_t = \min_i (\ell_i + 1)/k_i` is roughly where the Bessel functions start to oscillate. Without it, the error estimate can report convergence on subintervals spanning many decades in :math:`x` while the result is off by up to :math:`10^{-2}`. Wide intervals are split into such pieces from the start; the pieces count towards the maximum number of bisections. ``0`` disables it. **default: 5.0**

* ``non_oscillatory`` (``float``): Subintervals with :math:`(k_1 + k_2 + \dots)(b - a)` below this value use interpolatory quadrature on the collocation points instead of Levin's method, whose collocation matrix is nearly singular there. ``0`` disables it. **default: 0.5**


``get_integrand(x)``
'''''''''''

Calculates the all integrands passed to Levin (see ``levin``).

**Arguments:**

* ``x`` (1d ``numpy`` array): One dimensional array of values at which the integrand should be evaluated.

**Returns:**

* 2d ``numpy`` array of shape ``(len(x), N)``, where ``N``is the number of integrands passed in the constructor.


``get_quadrature()``
'''''''''''

Returns the stored integrals as quadrature rules, :math:`I_p = \sum_n w_n f(x_n)`, which hold for any integrand as long as ``x_min``, ``x_max``, ``k``, ``ell`` and ``integrand`` stay the same. This allows evaluating repeat calls elsewhere, for example with ``jax`` on a GPU, and with the integrand evaluated directly at the nodes instead of interpolated. ``update_integrand`` does not change the rules, so they only need to be exported once. Available after the second integration call (the one which stores the integrals, see the tutorial).

**Returns:**

* ``x`` (1d ``numpy`` array): Nodes :math:`x_n` of all integrals. ``levin`` itself sets the integrand to zero outside the support ``x`` passed in the constructor.

* ``weights`` (1d ``numpy`` array): Weights :math:`w_n`, same shape as ``x``.

* ``offsets`` (1d ``numpy`` array of ``integers``): The nodes of integral ``p`` are ``x[offsets[p]:offsets[p + 1]]``. If ``diagonal == True``, ``p`` is the variable index ``i`` of ``result[i]``; otherwise ``result[i, j]`` (variable ``i``, integrand ``j``) is ``p = j * M + i``.


``update_integrand(x, integrand, logx, logy)``
'''''''''''

Updates the integrand. CAVEAT: Only update the integrand if you DO NOT change anything in the oscillatory part of the integral, i.e. any argument of the Bessel function or the integration boundaries.
In other words, DO NOT use ``update_integrand`` if you change any of the following variables: ``x_min``, ``x_max``, ``k`` or ``ell``. This applies as well when calculating integrals over products of Bessel functions.
In the latter case you have to create a new instance of the class as ``levin`` will otherwise use precomputed quantities for previous settings (see also the tutorial for more details on this).
Repeat calls are fastest with ``logy = False`` and an unchanged ``x``: the integrals are then weighted sums of the new samples, without computing the spline.

**Arguments:**

* ``x`` (1d ``numpy`` array): One dimensional array over the support of the integrand :math:`x`.  Needs the same number of integrands as in the constructor.

* ``integrand`` (2d ``numpy`` array): Two dimensional array of the integrands :math:`f(x)`. Needs the same number of integrands as in the constructor.
        
* ``logx`` (``True`` / ``False``): Should the integrands be interpolated logarithmically in x or linearly?

* ``logy`` (``True`` / ``False``): Should the integrands, :math:`f(x)`,  be interpolated logarithmically in :math:`y = f(x)` or linearly, automatically checks for each integrand if this is possible.


``levin_integrate_bessel_single(x_min, x_max, k, ell, result, integrand = None)``
'''''''''''

Calculates integrals of the type:

.. math::

    I(k,\ell) = \int_a^b j_\ell(xk) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``0`` or ``1``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result``.

**Arguments:**

* ``x_min`` (1d ``numpy`` array): Values of the lower integration bound, :math:`a`. This array has shape ``(M)``.

* ``x_max`` (1d ``numpy`` array): Values of the upper integration bound, :math:`b`. This array has shape ``(M)``.

* ``k`` (1d ``numpy`` array): Values of the frequency in the Bessel function, :math:`k`. This array has shape ``(M)``.

* ``ell`` (1d ``numpy`` array of ``integers``): Values of the order of the Bessel function, :math:`\ell`. This array has shape ``(M)``.

* ``result`` (2 or 1d ``numpy`` array): This array needs to be defined before with the correct shape as it is passed by reference. If ``diagonal == False`` it must have the shape ``(M,N)``. If ``diagonal == True`` it must have shape ``(M)``.

* ``integrand`` (1d ``numpy`` array of ``integers``, only if ``diagonal == True``): Index of the integrand used by each variable, shape ``(M)``, so that ``result[i]`` is the integral over integrand ``integrand[i]``. This lets many variables share the same integrands, e.g. all multipoles for a set of integrands which depend on ``k`` only. If not given, ``M = N`` and variable ``i`` uses integrand ``i``. Like ``k`` and ``ell``, it must not change between calls.


``levin_integrate_bessel_double(x_min, x_max, k_1, k_2, ell_1, ell_2, result, integrand = None)``
'''''''''''

Calculates integrals of the type:

.. math::

   I(k_1, k_2,\ell_1,\ell_2) = \int_a^b j_{\ell_1}(xk_1)j_{\ell_2}(xk_2) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``2`` or ``3``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result``.

**Arguments:**

See the logic explained in ``levin_integrate_bessel_single``. ``ell_1``, ``k_1``, etc. must have the shapes as ``ell``, ``k`` above.


``levin_integrate_bessel_triple(x_min, x_max, k_1, k_2, k_3, ell_1, ell_2, ell_3, result, integrand = None)``
'''''''''''

Calculates integrals of the type:

.. math::

   I(k_1, k_2 ,k_3, \ell_1,\ell_2, \ell_3) = \int_a^b j_{\ell_1}(xk_1)j_{\ell_2}(xk_2) j_{\ell_3}(xk_3) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``4`` or ``5``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result``.

**Arguments:**

See the logic explained in ``levin_integrate_bessel_single``. ``ell_1``, ``k_1``, etc. must have the shapes as ``ell``, ``k`` above.

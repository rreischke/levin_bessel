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



``set_levin(n_col_in, maximum_number_bisections_in, relative_accuracy_in, super_accurate, verbose)``
'''''''''''

Sets up the internal parameters of the Levin integrator. If not called, default values are used.

**Arguments:**

* ``n_col_in`` (``integer``): Number of collocation points used at which the solution to the differential equation is constructed. **default: 8**

* ``maximum_number_bisections_in`` (``integer``): Maximum number of bisections used for the adaptive integration. **default: 32**

* ``relative_accuracy_in`` (``float``): Relative target accuracy at which the adaptive integration (the bisection) is stopped. **default: 1e-4**

* ``super_accurate`` (``True`` / ``False``): Should ``gsl`` (``False``)  or ``boost`` (``True``) be used for the computation of the Bessel functions. Especially at higher orders (larger than 100) of the Bessel functions ``gsl``can fail. **default: False**

* ``verbose`` (``True`` / ``False``): Should the code talk to you? **default: False**


``get_integrand(x)``
'''''''''''

Calculates the all integrands passed to Levin (see ``levin``).

**Arguments:**

* ``x`` (1d ``numpy`` array): One dimensional array of values at which the integrand should be evaluated.

**Returns:**

* 2d ``numpy`` array of shape ``(len(x), N)``, where ``N``is the number of integrands passed in the constructor.


``update_integrand(x, integrand, logx, logy)``
'''''''''''

Updates the integrand.

**Arguments:**

* ``x`` (1d ``numpy`` array): One dimensional array over the support of the integrand :math:`x`.  Needs the same number of integrands as in the constructor.

* ``integrand`` (2d ``numpy`` array): Two dimensional array of the integrands :math:`f(x)`. Needs the same number of integrands as in the constructor.
        
* ``logx`` (``True`` / ``False``): Should the integrands be interpolated logarithmically in x or linearly?

* ``logy`` (``True`` / ``False``): Should the integrands, :math:`f(x)`,  be interpolated logarithmically in :math:`y = f(x)` or linearly, automatically checks for each integrand if this is possible.


``levin_integrate_bessel_single(x_min, x_max, k, ell, diagonal, result)``
'''''''''''

Calculates integrals of the type:

.. math::

    I(k,\ell) = \int_a^b j_\ell(xk) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``0`` or ``1``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result`` and ``diagonal``

**Arguments:**

* ``x_min`` (1d ``numpy`` array): Values of the lower integration bound, :math:`a`. This array has shape ``(M)``.

* ``x_max`` (1d ``numpy`` array): Values of the upper integration bound, :math:`b`. This array has shape ``(M)``.

* ``k`` (1d ``numpy`` array): Values of the frequency in the Bessel function, :math:`k`. This array has shape ``(M)``.

* ``ell`` (1d ``numpy`` array of ``integers): Values of the order of the Bessel function, :math:`\ell`. This array has shape ``(M)``.

* ``diagonal`` (``True`` / ``False``): If ``M = N`` the code can be asked to only calculate the diagonal elements (``True``) of the ``(N,N)`` integrals.

* ``result`` (2 or 1d ``numpy`` array): This array needs to be defined before with the correct shape as it is passed by reference. If ``diagonal == False`` it must have the shape ``(M,N)``. If ``diagonal == True`` it must have shape ``(N)``.


``levin_integrate_bessel_double(x_min, x_max, k_1, k_2, ell_1, ell_2, diagonal, result)``
'''''''''''

Calculates integrals of the type:

.. math::

   I(k_1, k_2,\ell_!,\ell_2) = \int_a^b j_{\ell_1}(xk_1)j_{\ell_2}(xk_2) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``2`` or ``3``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result`` and ``diagonal``

**Arguments:**

See the logic explained in ``levin_integrate_bessel_single``. ``ell_1``, ``k_1``, etc. must have the shapes as ``ell``, ``k`` above.


``levin_integrate_bessel_triple(x_min, x_max, k_1, k_2, k_3, ell_1, ell_2, ell_3, diagonal, result)``
'''''''''''

Calculates integrals of the type:

.. math::

   I(k_1, k_2, ,k_3, \ell_!,\ell_2, \ell_3) = \int_a^b j_{\ell_1}(xk_1)j_{\ell_2}(xk_2) j_{\ell_3}(xk_3) f(x) \mathrm{d}x

where :math:`f(x)` are the integrands and :math:`j_\ell(x)` can be spherical or cylindrical Bessel functions. ``type`` in ``levin`` needs to be set to ``4`` or ``5``. Generally, if you have specified ``N`` integrands before, this function can be passed ``M`` variables, so that
in the end ``(M, N)`` integrals are calculated. For the specifics see ``result`` and ``diagonal``

**Arguments:**

See the logic explained in ``levin_integrate_bessel_single``. ``ell_1``, ``k_1``, etc. must have the shapes as ``ell``, ``k`` above.



**Stuff**

.. automodule:: pylevin
   :members:
   :imported-members:
   :show-inheritance:
   :special-members:

.. automodule:: pylevin.set_levin
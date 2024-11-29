def levin(type, x, integrand, logx, logy, nthread):
    """
    Initialises the Levin integrator class given the type of Bessel function 
    and if it is a product or not, reads in the integrands and their support
    and wether they should be interpolated logarithmically or not.

    Args:
    ----------
        type (integer): Specifies the type of integral you want to calculate.
                        Currently, the following types are implemented:

                            * 0: single spherical Bessel function
                            * 1: single cylindrical Bessel function
                            * 2: double spherical Bessel function
                            * 3: double cylindrical Bessel function
                            * 4: triple spherical Bessel function
                            * 5: triple cylindrical Bessel function
        
        x    (1d array):    One dimensional array over the support of the integrand :math:`x`.
                            The integration used later when integrating should be within the
                            minimum and maximum value of this array.

        integrand (2d array) Two dimensional array of the integrands :math:`f(x)`. Needs at least
                            the shape ``len(x), 1``. Generally it should have the shape ``len(x), N``,
                            where ``N``is the number of different integrals you want to calculate.
                            In other words, you can pass a series of different :math: `f(x)`.
                            
    """
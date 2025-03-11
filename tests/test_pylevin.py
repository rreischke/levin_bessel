import pylevin as levin
import numpy as np
import pytest


def test_levin_integrate_bessel_single():
    x_length = 100
    N = 2
    x = np.geomspace(1e-5,100,x_length) #define support
    y = np.linspace(1,2, N) 
    f_of_x = x[:,None]**(3*y[None,:]) + (x**2 +x)[:, None] #define integrands f(x) 
    integral_type = 0 
    N_thread = 1 # Number of threads used for hyperthreading
    logx = True # Tells the code to create a logarithmic spline in x for f(x)
    logy = True # Tells the code to create a logarithmic spline in y for y = f(x)
    lp_single = levin.pylevin(integral_type, x, f_of_x, logx, logy, N_thread) #Constructor of the class

    n_sub = 10 #number of collocation points in each bisection
    n_bisec_max = 32 #maximum number of bisections used
    rel_acc = 1e-4 #relative accuracy target
    boost_bessel = True #should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
    verbose = False #should the code talk to you?
    lp_single.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
    M = 1000 #number of arguments at which the integrals are evaluated
    k = np.geomspace(1e-3,1e4,M)
    ell = (5*np.ones_like(k)).astype(int) #order of the Bessel function, needs to be an integer
    result_levin = np.zeros((M, N)) #allocate the result
    lp_single.levin_integrate_bessel_single(x[0]*np.ones_like(k), x[-1]*np.ones_like(k), k, ell, result_levin)


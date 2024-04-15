import levinpower
import numpy as np
import time

x = np.geomspace(1e-5,100,100) #define support
f_of_x = (x**3 -x**2 +x)[:, None]*x[None, :] #define f(x) 
integral_type = 4 # define integral type for Levin, 4 correpsonds to double spherical Bessel function
N_thread = 8 # Number of threads used for hyperthreading
logx = True # Tells the code to create a logarithmic spline in x for f(x)
logy = True # Tells the code to create a logarithmic spline in y for y = f(x)
lp = levinpower.levinpower(integral_type, x, f_of_x, logx, logy, N_thread) #Constructor of the class
lp.set_levin(8,32,1e-6,False,False)

N = 10
k = np.geomspace(1e-2,1000,N)
n_order_1 = 5
n_order_2 = 10
n_order_3 = 15
ell_1 = (n_order_1*np.ones_like(k)).astype(int)
ell_2 = (n_order_2*np.ones_like(k)).astype(int)
ell_3 = (n_order_2*np.ones_like(k)).astype(int)

t0 = time.time()
result_levin = np.array(lp.levin_integrate_bessel_triple(x[0]*np.ones_like(k),x[-1]*np.ones_like(k),k,k,k,ell_1,ell_2,ell_3))
print("Levin took", time.time() -t0, "s")

t0 = time.time()
result_levin_rerun = np.array(lp.levin_integrate_bessel_triple(x[0]*np.ones_like(k),x[-1]*np.ones_like(k),k,k,k,ell_1,ell_2,ell_3))
print("Levin took", time.time() -t0, "s")

import pyccl as ccl
import pylevin as levin
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson


cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.965)

z = np.linspace(0.55, .65, 100)
nz = np.exp(-0.5*((z-.6)/.01)**2)
ell = np.unique((np.geomspace(2, 400)).astype(int))


chi = ccl.comoving_radial_distance(cosmo,1/(1+z))
kmin, kmax, nk = 1e-4, 1e1, 500
k = np.geomspace((kmin), (kmax), nk) # Wavenumber
pk_nl = []
for zet in z:
    pk_nl.append(ccl.nonlin_matter_power(cosmo, k, 1/(1+zet)))
pk_nl = np.array(pk_nl)


spl = UnivariateSpline(chi, z, k=2, s=0)
dzdchi = spl.derivative()(chi)
norm = 1/np.trapz(nz*dzdchi,chi)
nofchi = dzdchi*nz*norm
idx_non_zero = np.where(nofchi >0)[0]


integral_type = 0
N_thread = 1 # Number of threads used for hyperthreading
logx = True # Tells the code to create a logarithmic spline in x for f(x)
logy = True # Tells the code to create a logarithmic spline in y for y = f(x)
n_sub = 6 #number of collocation points in each bisection
n_bisec_max = 4 #maximum number of bisections used
rel_acc = 2e-3 #relative accuracy target
boost_bessel = True #should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
verbose = False #should the code talk to you?

lower_limit = k[0]*np.ones_like(ell)
upper_limit = np.ones_like(ell)

N_int = int(1e3)
k_int = np.geomspace(k[0], 1e0, N_int)

inner_int = np.zeros((len(ell), len(k_int)))
pk_nl_new = np.zeros((len(z), len(k_int)))
for zet_i, zet_val in enumerate(z):
    pk_nl_new[zet_i,:] = (ccl.nonlin_matter_power(cosmo, k_int, 1/(1+zet_val)))

lp_ell = []
lower_limit = chi[idx_non_zero[0]]*np.ones_like(k_int)
upper_limit = chi[idx_non_zero[-1]]*np.ones_like(k_int)

t0 = time.time()
for i_ell, val_ell in enumerate(ell):
    lp = levin.pylevin(integral_type, chi[idx_non_zero], np.sqrt(pk_nl_new[idx_non_zero,:])*nofchi[idx_non_zero,None], logx, logy, N_thread, True)
    lp_ell.append(lp)
    lp_ell[i_ell].set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
    ell_values = (val_ell*np.ones_like(k_int)).astype(int)
    lp_ell[i_ell].levin_integrate_bessel_single(lower_limit, upper_limit, k_int, ell_values, inner_int[i_ell,:])
print("Levin took", time.time() -t0, "s")
t0 = time.time()
for i_ell, val_ell in enumerate(ell):
    ell_values = (val_ell*np.ones_like(k_int)).astype(int)
    lp_ell[i_ell].levin_integrate_bessel_single(lower_limit, upper_limit, k_int, ell_values, inner_int[i_ell,:])
print("Levin took", time.time() -t0, "s")

ell_values = (ell[:,None]*np.ones_like(k_int)[None,:]).astype(int)

t0 = time.time()

for i_ell, val_ell in enumerate(ell):
    lp_ell[i_ell].levin_integrate_bessel_single(lower_limit, upper_limit, k_int, ell_values[i_ell,:], inner_int[i_ell,:])
print("Levin took", time.time() -t0, "s")

result_levin_second = 2/np.pi*simpson(inner_int**2*k_int**2,x = k_int, axis = -1)
    




fontsi = 20
fontsi2 = 12


fig, ax = plt.subplots(1)
import matplotlib as mpl
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('image', interpolation='none')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

ax.loglog(ell,result_levin_second, ls = "--", label = r"$\mathrm{\bf{pylevin}} $", lw = 2, color = "red")

ax.set_ylim(1.1*np.min(result_levin_second),1.1*np.max(result_levin_second))
ax.set_xlim(ell[0],ell[-1])
ax.set_xticks([])

ax.set_xlabel(r"$\ell$", fontsize = fontsi)
ax.set_ylabel(r"$C_\ell$", fontsize = fontsi)
ax.legend(fontsize = fontsi2, loc = 'lower left', ncols = 3, frameon=True)
ax.tick_params(axis='both', which='major', labelsize=fontsi2) 


plt.show()






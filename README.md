Implements Levin's method for products of spherical and cylindrical Bessel functions.

## Installation
For starters you first clone the directory via:
```shell
git clone git@github.com:rreischke/levin_bessel.git
```

Then navigate to the cloned directory
```shell
cd levin_bessel
conda env create -f conda_env.yaml
conda activate levinpower_env
pip install .
```
On some Linux servers you will have to install ``gxx_linux-64`` by hand and the installation will not work. This usually shows the following error message in the terminal:
``
gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
``
If this is the case just install it by typing
```shell
 conda install -c conda-forge gxx_linux-64
```
and redo the ``pip`` installation.

If you do not want to use the conda environment make sure that you have ``boost`` and ``gsl`` installed.
You can install both via ``conda``:
```shell
conda install -c conda-forge gsl
conda install -c conda-forge gxx_linux-64
conda install conda-forge::boost
git clone git@github.com:rreischke/levin_bessel.git
cd levin_bessel    
pip install .
```

## Examples
There are three notebooks in the ``test`` directory for the case of a single Bessel function and products of two or three Bessel functions. After installing you can quite generally use the code as follows:
```python 
import levinpower
N_thread = 4
integral_type = 4 
```

Here you import the library, define the number of threads for hyperthreading and define the integral type which itself can take six values:
```
- 0: single spherical Bessel function
- 1: single cylindrical Bessel function
- 2: double spherical Bessel function
- 3: double cylindrical Bessel function
- 4: triple spherical Bessel function
- 5: triple cylindrical Bessel function
```

Next it is time to define the integrands ``f_of_x`` over their support ``x`` and to decide whether they should be logarithmically or linearly interpolated in $x$ and $f(x)$. Then pass all of them to the class constructor
```python 
import numpy as np
x = np.geomspace(1e-5,100,100) 
f_of_x = (x**3 -x**2 +x)[:, None]*x[None,:]
logx = True 
logy = True 
lp = levinpower.levinpower(integral_type, x, f_of_x, logx, logy, N_thread)
```

It should be noted that the code always requires ``f_of_x`` to be a two-dimensional numpy array. So one can pass many integrands at the same time via the second index. If there is only a single integrand to pass, one should always pass ``f_of_x[:, None]``.
Next you can explicitely set the accuracy values of the integrator via
```python
n_sub = 8 
n_bisec_max = 16
rel_acc = 1e-6
boost_bessel = False
verbose = False
lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
```

If you do not pass any of those, default values will be assumed. Which are:
- ``n_sub = 8``, the number of collocation points for the Levin collocation
- ``n_bisec_max = 32``, maximum number of bisections used
- ``rel_acc = 1e-6``, relative accuracy after which the bisection is terminated
- ``boost_bessel = False``, if boost should be used to calculate the Bessel functions (this might be more accurate at higher Bessel order, but also slower). If ``False``, the ``gsl`` is used.
- ``verbose = False``, whether the code should tak to you. This obviously slows things down and should only be used if problems occur.

Now you can finally calculate an integral over a finite range of the specified type via
```python
N = 1000
k = np.geomspace(1e-3,100,N)
order = 5
ell = (order*np.ones_like(k)).astype(int)
result = np.array(lp.levin_integrate_bessel_single(x[0]*np.ones_like(k), x[-1]*np.ones_like(k), k, ell, False)) 
```

Here we evaluate the previously defined 100 integrands at 1000 arguments of the Bessel function each. If change the integral inly slightly (so that the required bisection would not change), as in many inference problems, you can simply update the integrand via
```python
lp.update_integrand(x,y, logx, logy)
```
This will speed up subsequent evaluation significantly. After reading this, please progress to the example notebooks in ```test``` and go through them starting with ```test_levin_single.ipynb```.




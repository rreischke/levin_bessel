---
  title: 'pylevin: efficient numerical integration of integrals containing up to three Bessel functions'
  tags:
    - Python
    - numerical integration
    - oscillatory functions
  authors:
    - name: Robert Reischke
      affiliation: 1
      orcid: 0000-0001-5404-8753
      corresponding: true
  
  affiliations:
    - name: Argelander Institut fuer Astronomie
      index: 1


  date: 24 February 2025
  bibliography: paper.bib
---

# Summary
Bessel functions naturally occur in physical systems with some degree of rotational symmetry. Theoretical predictions of observables therefore often involve integrals over those functions which are not solvable analytically and have to be treated numerically instead. However, standard integration techniques like quadrature generally fail to solve these types of integrals efficiently and reliably due to the very fast oscillations of the Bessel functions. Providing general tools to quickly compute these types of integrals is therefore paramount. `pylevin` can calculate the following types of frequently encountered integrals

$$
I_{\ell_1\ell_2\ell_3}(k_1,k_2,k_3) = \int_{a}^{b} \mathrm{d}x\,f(x) \prod_{i=1}^N \mathcal{J}_{\ell_i}(k_ix)\,,\quad N= 1,2,3\,,
$$

here $\mathcal{J}_\ell(x)$ denotes a spherical or cylindrical Bessel function of order $\ell$ and $f(x)$ can be any non-oscillatory function, i.e. with frequencies much lower than the one of the product of Bessel functions. 

# Statement of need
Typical approaches numerically estimate integrals over highly-oscillatory integrands are based on Fast Fourier Transforms (FFTLog) [@schoneberg_2018;@grasshorn_2018;@fang_2020] and asymptotic expansions [@levin_1996;@iserles_efficient_2005]. In `pylevin`, we implement one of the former methods, in particular, the adaptive Levin collocation [@levin_1996;@chen_2022;@leonard_2023]. Extending and improving the work done in @zieser_2016, `pylevin` can solve integrals of the type $I_{\ell_1\ell_2\ell_3}(k_1,k_2,k_3)$ (see summary). 

 The main code is implemented in `C++` and wrapped into `python` using `pybind`. Due to the way `pylevin` implements Levin's method it makes extensive use of precomputed quantities allowing updating the function $f(x)$ and making successive calls of the integration routine an order of magnitude faster than the first call. An aspect that is particularly important for situations where the same type of integral needs to be evaluated many times for slightly different $f(x)$. This is for example the case in inference when running Markov Chain Monte-Carlo.

In contrast to other implementations for highly oscillatory integrals, `pylevin` is very flexible, as it is not hardcoded and tailored to one particular application but is completely agnostic regarding the integrand. Furthermore, it implements integrals over three Bessel functions for the first time. These are for example required in many cosmological applications for higher-order statistics. Due to its implementation in a statically typed compiled language, it is also extremely fast, while making use of the convenience and white-spread use of `python` via `pybind`. 



# Examples
The way `pylevin` works is that one first defines an integrand, $f(x)$, the integral type (spherical or cylindrical Bessel functions and $N$) and if the interpolation of the integrand should be carried: out logarithmically

```python
x = np.geomspace(1e-5,100,100)
f_of_x = x**3 + (x**2 +x)
integral_type = 0 
number_omp_threads = 1 
interploate_logx = True
interploate_logy = True
lp_single = levin.pylevin(integral_type,
                          x,
                          f_of_x[:, None],
                          logx,
                          logy,
                          number_omp_threads)
```

Note that the broadcasting of `f_of_x`is required as one can in principle pass many different integrands at the same time and the code always expects this dimension. We can then define the values $k$ and $\ell$ at which we want to evaluate the integral which are all one-dimensional arrays of the same shape. Additionally, we also have to allocate the memory for the result which is stored in-place:

```python
k = np.geomspace(1e-3,1e4,1000)
ell = (5*np.ones_like(k)).astype(int) 
result_levin = np.zeros((len(k), 1)) 
lp_single.levin_integrate_bessel_single(x[0]*np.ones_like(k),
                                        x[-1]*np.ones_like(k),
                                        k,
                                        ell,
                                        False,
                                        result_levin)
```

If we would have passed more integrands before, the results must have the corresponding shape in the second dimension. For more detailed examples we refer to the example notebook on github and to the API.

 We now demonstrate the performance of `pylevin` on a single core on an Apple M3 and compare it to `scipy.integrate.quad`, an adaptive quadrature. The relative accuracy required for both methods is set to $10^{-3}$.
We use the following two integrals as an example:

$$
I_2 = \;\int_{10^{-5}}^{100} \mathrm{d}x \;(x^3 +x^2 +x)j_{10}(kx)j_5(kx)\;, 
$$
$$
I_3 = \;\int_{10^{-5}}^{100} \mathrm{d}x \;(x^3 +x^2 +x)j_{10}(kx)j_5(kx)j_{15}(kx)\;,
$$

The result of $I_2$ is shown on the left and for $I_3$ on the right in \autoref{fig:figure1}. In order for the quadrature to converge over an extended $k$-range, the number of maximum sub-intervals was increased to $10^3$ ($2\times 10^3$) for $I_2$ ($I_3$). The grey-shaded area indicates where the quadrature fails to reach convergence even after this change. 
It is therefore clear that `pylevin` is more accurate and around three to four orders of magnitudes faster than standard integration routines. 

![Speed and accuracy comparison of `pylevin` (shown in dashed red) against a standard adaptive quadrature (shown in solid blue). The runtime for the two methods is given in the legend. For the adaptive quadrature the maximum number of sub-intervals was set to 1000 (default is 50). The grey shaded region indicates when the quadrature starts to fail. The bottom panel shows the relative difference between the two methods.   **Left**: Result of the integral $I_2$. **Right**: Result of the integral $I_3$.  \label{fig:figure1}](paper_plot_joss.pdf)


# Comparison with various codes
Additionally to the benchmark, `pylevin` is compared with more specialised, codes who mostly solve integrals over single Bessel functions. All computing times presented here are averages over several runs. The comparison was done with an M3 processor with 8 cores (4 performance cores).

## `hankel`
We calculate the following Hankel transformation with `pylevin` and Ogata's method (@ogata_2005) as implemented in the `hankel` package (@murray_2019).

$$
\mathrm{integral}(k) = \int_0^\infty\frac{x^2}{x^2+1}J_0(kx)\;\mathrm{d}x\;,
$$

for 500 values of $k$ logarithmically-spaced between 1 and $10^4$. The result is depicted on the left side of \autoref{fig:figure2}. It can be seen that both methods agree very well and are roughly equally fast. While the Hankel transformation formally goes from zero to infinity, $a=10^{-5}$ and $b = 10^8$ was used for `pylevin`. This choice of course depends on the specific integrand. 

## `hankl`
Here, we follow the cosmology example provided in the `hankl` documentation (@karamanis_2021) to compute the monopole of the galaxy power spectrum:

$$
\xi_0(s) =\int_0^\infty\left(b^2+fb/3 +f^2/5\right)P_\mathrm{lin}(k)J_0(ks) k^2\;\mathrm{d}k\;,
$$

where $b$ is the galaxy bias, $f$ the logarithmic growth rate and $P_\mathrm{lin}(k)$ is the linear matter power spectrum, which is calculated using `camb` (@lewis_cosmological_2002) at six redshifts. Since `hankl` is FFT based, it requires $k$ to be discretised, the FFT-dual will then be calculated at the inverse grid points. For this comparison, we use $2^{10}$ logarithmically-spaced points between $k=10^{-4}$ and $k = 1$ for the transformation to converge. For `pylevin`, the number of points where the transformation is evaluated is arbitrary. Here we use 100 points, which is more than enough to resolve all features in $\xi_0$.
The results are shown on the right of \autoref{fig:figure2} and good agreement can be found between the two methods with `hankl` being roughly twice as fast as `pylevin`. 

![Comparison of `pylevin` with two methods to calculate a Hankel transformation. Dashed red is `pylevin` while solid blue is the alternative method. **Left**: Integral$(k)$ evaluated with the Ogata method using the `hankel` package. **Right**: Integral for the galaxy power spectrum monopole evaluated using the **hankl** package. Different lines refer to different redshifts.  \label{fig:figure2}](paper_plot_2_joss.pdf)


# References
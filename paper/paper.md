---
  title: 'pylevin: efficient numerical integration of integrals containing up to three Bessel functions'
  tags:
    - Python
    - numerical integration
    - oscillatory functions
  authors:
    - name: Robert Reischke
      orcid: 0000-0001-5404-8753
      corresponding: true
      affiliation: 1
  
  affiliations:
    - name: Argelander Institut fuer Astronomie
    - index: 1
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

Additionally to this benchmark, `pylevin` was tested against some specialised methods estimating integrals including a single Bessel function such as Ogata's method @ogata_2005 with an implementation described in @murray_2019 or FFTLog-based methods [@hamilton_2000;@karamanis_2021;@leonard_2023] and found excellent agreement in the results and runtimes within a factor of two of the other methods.



# References
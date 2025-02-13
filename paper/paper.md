---
  title: 'pylevin: efficient numerical integration of integrals over products of up to three Bessel functions'
  tags:
    - Python
    - numerical integration
    - oscillatory functions
  authors:
   - name: Robert Reischke
     orcid: 0000-0001-5404-8753
     affiliation: Argelander Institut für Astronomie
  date: 3 December 2024
  bibliography: paper.bib
---

# Summary
Bessel functions naturally occur in physical systems with some degree of rotational symmetry. Theoretical predictions of observables therefore often involve integrals over those functions which are not solvable analytically and have to be treated numerically instead. However, standard integration techniques like quadrature generally fail to solve these types of integrals efficiently and reliably due to the very fast oscillations of the Bessel functions. Providing general tools to quickly compute these types of integrals is therefore paramount. `pylevin` can calculate the following types of frequently encountered integrals

$$
I_{\ell_1\ell_2\ell_3}(k_1,k_2,k_3) = \int_{a}^{b} \mathrm{d}x\,f(x) \prod_{i=1}^N j_{\ell_i}(k_ix)\,,\quad N= 1,2,3\,,
$$

here $j_\ell(x)$ denotes a spherical or cylindrical Bessel function of order $\ell$ and $f(x)$ can be any non-oscillatory function, i.e. with frequencies much lower than the one of the product of Bessel functions.


# Statement of need
Typical approaches numerically estimate integrals over highly-oscillatory integrands are based on Fast Fourier Transforms (FFTLog) [@schoneberg_2018;@grasshorn_2018;@fang_2020] and asymptotic expansions [@levin_1996;@iserles_efficient_2005]. In `pylevin`, we implement one of the former methods, in particular, the adaptive Levin collocation [@levin_1996;@chen_2022;@leonard_2023]. Extending and improving the work done in @zieser_2016, `pylevin` can solve integrals of the type $I_{\ell_1\ell_2\ell_3}(k_1,k_2,k_3)$ (see summary). 

 The main code is implemented in `C++` and wrapped into `python` using `pybind`. Due to the way `pylevin` implements Levin's method it makes extensive use of precomputed quantities allowing updating the function $f(x)$ and making successive calls of the integration routine an order of magnitude faster than the first call. An aspect that is particularly important for situations where the same type of integral needs to be evaluated many times for slightly different $f(x)$. This is for example the case in inference when running Markov Chain Monte-Carlo.

In contrast to other implementations for highly oscillatory integrals, `pylevin` is very flexible, as it is not hardcoded and tailored to one particular application but is completely agnostic regarding the integrand. Furthermore, it implements integrals over three Bessel functions for the first time. These are for example required in many cosmological applications for higher-order statistics. Due to its implementation in a statically typed compiled language, it is also extremely fast, while making use of the convenience and white-spread use of `python` via `pybind`. 


As an example, we show the performance of `pylevin` on a single core on an Apple M3 and compare it to `scipy.integrate.quad`, an adaptive quadrature. The relative accuracy required for both methods is set to $10^{-3}$.
We use the following integral as an example:

$$
\int_{1\times 10^{-5}}^{100}\mathrm{d}x (x^3 + x^2 + x) j_5(xk)j_{10}(xk)\,.
$$

The results are shown in \autoref{fig:figure}

![Top panel: Result of the integral times $k^2$ to highlight the high frequency regime. The quadrature is shown in solid blue and `pylevin` in dashed red. The runtime for the two methods is given in the legend. For the adaptive quadrature the maximum number of sub-intervals was set to 1000 (default is 50). The grey shaded region indicates when the quadrature starts to fail. Bottom panel: relative difference between the two methods.  \label{fig:figure}](paper_plot_two_bessel.pdf)



# References
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
  date: 23 February 2025
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

![Top panel: Result of the integral, $I_2$, times $k^2$ to highlight the high frequency regime. The quadrature is shown in solid blue and `pylevin` in dashed red. The runtime for the two methods is given in the legend. For the adaptive quadrature the maximum number of sub-intervals was set to 1000 (default is 50). The grey shaded region indicates when the quadrature starts to fail. Bottom panel: relative difference between the two methods.  \label{fig:figure1}](paper_plot_two_bessel.pdf)

![Same as \autoref{fig:figure1} but for the integral, $I_3$ times $k^3$.  \label{fig:figure2}](paper_plot_three_bessel.pdf){width=50%}

As an example, we show the performance of `pylevin` on a single core on an Apple M3 and compare it to `scipy.integrate.quad`, an adaptive quadrature. The relative accuracy required for both methods is set to $10^{-3}$.
We use the following two integrals as an example:

$$
I_2 = \;\int_{10^{-5}}^{100} \mathrm{d}x \;(x^3 +x^2 +x)j_{10}(kx)j_5(kx)\;, 
$$
$$
I_3 = \;\int_{10^{-5}}^{100} \mathrm{d}x \;(x^3 +x^2 +x)j_{10}(kx)j_5(kx)j_{15}(kx)\;,
$$

The result of $I_2$ is shown in \autoref{fig:figure1} and for $I_3$ in \autoref{fig:figure2}. In order for the quadrature to converge over an extended $k$-range, the number of maximum sub-intervals was increased to $10^3$ ($2\times 10^3$) for $I_2$ ($I_3$). The grey-shaded area indicates where the quadrature fails to reach convergence even after this change. 
It is therefore clear that `pylevin` is more accurate and around three to four orders of magnitudes faster than standard integration routines. 

Additionally to this benchmark, `pylevin` was tested against some specialised methods estimating integrals including a single Bessel function such as Ogata's method @ogata_2005 with an implementation described in @murray_2019 or FFTLog-based methods [@hamilton_2000;@karamanis_2021;leonard_2023] and found excellent agreement in the results and runtimes within a factor of two of the other methods.



# References
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
Bessel functions naturally occur in physical systems with some degree of rotational symmetry. Derived quantities for these physical models often involve integrals over those functions which are not solvable analytically and have to be treated numerically instead. However, standard integration techniques like quadrature generally fail to solve these types of integrals efficiently and reliably due to the very fast oscillations of the Bessel functions. Providing general tools to quickly compute these types of integrals is therefore paramount.


# Statement of need
Typical approaches numerically estimate integrals over highly-oscillatory integrands are based on Fast Fourier Transforms (FFTLog) and asymptotic expansions [@iserles_efficient_2005], `pylevin` implements the latter method 


# References
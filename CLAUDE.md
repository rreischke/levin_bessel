# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pylevin` is a C++20 library (exposed to Python via pybind11) that evaluates oscillatory integrals containing products of up to three spherical/cylindrical Bessel functions using Levin's method (Chebyshev collocation). Paper: arXiv:2502.12142. Docs: levin-bessel.readthedocs.io (sources in `docs/`: usage guide in `docs/guides/running.rst`, API in `docs/api.rst`).

## Build & test

- Dependencies: GSL, Boost, a C++20 compiler with OpenMP (on macOS, Apple clang with `libomp`; see `setup.py`). `conda_env.yaml` sets up the environment.
- The compiled extension is a single module `pylevin`; **there is no pure-Python code**. Any change to `src/*` needs a rebuild before tests see it. The in-place build `pylevin.cpython-312-darwin.so` in the repo root shadows the installed one when running from the repo root.
- **Build with the target environment's own compiler settings.** On this machine the shell has base conda's compiler variables (`CC` = conda clang 14, `LDFLAGS` with `-L/-rpath /opt/anaconda3/lib`). Builds then link base's libomp 18, which lacks `__kmpc_dispatch_deinit` that Apple clang ≥ 19 emits, and the import fails in `levinpower_env` (`symbol not found in flat namespace`). Build and install for `levinpower_env` (libomp 19) with:

  ```bash
  P=/opt/anaconda3/envs/levinpower_env
  env -u CFLAGS -u CXXFLAGS CC=clang CXX=clang++ CPPFLAGS="-isystem $P/include" LDFLAGS="-L$P/lib -Wl,-rpath,$P/lib" $P/bin/python setup.py build_ext --inplace  # or: -m pip install .
  rm -rf build
  ```

- **Delete `build/` after every build.** `pip install .` reuses an up-to-date extension from `build/`, even one linked against another environment.
- Tests: `$P/bin/python -m pytest` (all), `pytest tests/test_repeat_calls.py` (repeat calls, `get_quadrature`, integrand map, spline, argument errors). `tests/cells.py` and `tests/non_limber.py` are scripts, not pytest tests.
- `setup.py` compiles with `-O3 -ffast-math`; keep this in mind for NaN/inf handling and floating-point reproducibility.

## Architecture

All logic lives in one class, `pylevin` ([src/pylevin.h](src/pylevin.h), [src/pylevin.cpp](src/pylevin.cpp)). [src/pybind11_interface.cpp](src/pybind11_interface.cpp) only binds it (docstrings for the Python API live there; `set_levin`, `update_integrand` and the integration calls release the GIL, `get_integrand` and `get_quadrature` create numpy arrays and keep it). Python arguments are `array_double` / `array_int` (`py::array_t` with `c_style | forcecast`), read in place; do not reintroduce `std::vector` arguments, which pybind11 converts element by element.

Usage flow: construct `pylevin(type, x, integrand, logx, logy, nthread, diagonal)` → `set_levin(...)` → `levin_integrate_bessel_{single,double,triple}(x_min, x_max, k..., ell..., result, integrand=None)`. `type` selects the Bessel kind/combination. In diagonal mode, variable `i` uses integrand `integrand[i]` (identity if not given); otherwise every variable uses every integrand.

Integrands: an own not-a-knot cubic spline (identical to `scipy.interpolate.CubicSpline`) on `x_integrand` (log x if `logx`), of y or log y (`is_y_log`, per integrand, only if positive everywhere). `factor_spline_grid` factorises the tridiagonal system once per grid; `set_spline_coefficients` solves it per integrand (parallel in `init_splines`, scratch passed in). `inhomogeneity` evaluates the spline with a per-thread segment cache and returns 0 outside the grid. Akima/GSL splines are gone.

Core algorithm (per integral, per interval `[A, B]`):
- `solve_LSE_{single,double,triple}` build the Levin collocation system with Chebyshev basis (`n_col` nodes, `setNodes_cheby`, `basis_function_cheby`), with ODE coefficients from `fill_w_*` / `A_matrix_*` (the Bessel recursion relations, which is why new integral types need a derivative recursion relation).
- `integrate_*` evaluates the result from the solved coefficients; `iterate_*` bisects adaptively until the relative/absolute tolerance is met. `iterate_pieces` is the shared driver.
- Accuracy safeguards in `set_levin` (`max_ratio`, `non_oscillatory`): wide intervals are pre-split by ratio `b / max(a, x_t)`, and low-oscillation pieces use interpolatory quadrature (`smooth_integral`) because the collocation matrix is near-singular there. Setting both to 0 reproduces older behaviour.
- Single-Bessel has a banded Chebyshev solver (`solve_LSE_single_banded`, `use_banded_levin`, default for `n_col` ≥ 64) with caches (`BandedSystemCache`, `DenseFallbackCache`, mutex-protected) and a dense LU fallback.

Repeat calls (the main performance path):
1. First call: bisection only (`bisection`).
2. Second call (storing): every piece gets quadrature weights `weights_precomp` with integral = Σ_j w_j f(x_j): `store_levin_weights` (two transposed triangular solves on the LU the dense solver leaves; the storing call always uses the dense solver, also when banded) or the interpolatory weights of `smooth_integral`. `build_grid_tables` then builds, for linear y, either `sample_weights` (moments folded with the spline by `fold_spline_moments`, the adjoint of `set_spline_coefficients`, trimmed to the touched segments ± 64 knots) or `spline_moments`, whichever is cheaper; for log y, `grid_nodes` (segment, t, weight per node).
3. Later calls: `integrate_lse_set` uses `sample_weights` → `spline_moments` → `grid_nodes` → `weights_precomp` (fallback, e.g. after a grid change). With `sample_weights`, `update_integrand` only stores `spline_samples` and defers the linear-y coefficients (`spline_coefficients_stale`).

Invariants: stored data is indexed by `precomp_index` (the variable in diagonal mode, `integrand * size_variables + variable` otherwise). A new grid must clear `spline_moments`, `grid_nodes` and `sample_weights`; `set_levin` clears all stored data. Everything that evaluates the spline after an update must call `ensure_spline_coefficients` first (the bisection/storing calls and `get_integrand` do). Every repeat path must agree with the rule from `get_quadrature` applied to the spline, which the tests check.

Parallelism: OpenMP over (k, ell) evaluations with per-thread scratch buffers (`*_set` vectors indexed by thread id). New hot-path code should use these buffers, not heap allocation; shared caches must be locked; `is_y_log` is a `std::vector<bool>` and must not be written in parallel.

## Other directories

- `paper/` – JOSS paper and plot notebook; `tutorial/levin_tutorial.ipynb` – user tutorial (including the integrand map, `get_quadrature` with `jax`, and the batched matrix form). Changes to the public API should update `docs/api.rst`, `docs/guides/running.rst`, the docstrings in `pybind11_interface.cpp`, and the tutorial.
- `.github/workflows` only builds the paper PDF; there is no CI for the code.

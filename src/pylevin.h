#ifndef PYLEVIN_H
#define PYLEVIN_H

#include <vector>
#include <array>
#include <optional>
#include <numeric>
#include <omp.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <thread>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <functional>

class pylevin
{
public:
  // Numpy arguments are read in place (or converted once, in C); no element-wise conversion.
  using array_double = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;
  using array_int = pybind11::array_t<std::int64_t, pybind11::array::c_style | pybind11::array::forcecast>;

private:
  struct BandedSystemCache
  {
    bool valid = false;
    std::vector<double> original_matrix;
    std::vector<double> factored_matrix;
    std::vector<std::vector<double>> corrections;
    std::vector<double> boundary_matrix;
  };
  struct DenseFallbackCache
  {
    gsl_matrix *lu = nullptr;
    gsl_permutation *permutation = nullptr;
    size_t bytes = 0;
    uint64_t last_used = 0;

    ~DenseFallbackCache()
    {
      if (lu) gsl_matrix_free(lu);
      if (permutation) gsl_permutation_free(permutation);
    }
  };
  using BandedCacheKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                                    uint, uint, uint>;
  struct SplineMoment
  {
    uint segment;
    double moment[4];
  };
  // Last grid segment found by each thread (one cache line each).
  struct alignas(64) SegmentCache
  {
    uint segment = 0;
  };

  // Support of the integrands, in log x if is_x_log.
  std::vector<double> x_integrand;
  // Grid spacing and factorised not-a-knot spline system of x_integrand (factor_spline_grid).
  std::vector<double> spline_h, spline_inverse_h, spline_upper, spline_multiplier, spline_inverse_pivot;
  std::vector<SegmentCache> segment_cache;
  std::vector<bool> is_y_log;
  std::vector<uint> index_variable, index_bisection;
  std::vector<std::vector<double>> bisection;
  std::vector<gsl_vector *> F_stacked_set, F_stacked_set_half;
  std::vector<gsl_vector *> ce_set, ce_set_half;
  std::vector<gsl_matrix *> matrix_G_set, matrix_G_set_half;
  std::vector<gsl_permutation *> P_set, P_set_half;
  std::vector<std::vector<double>> x_j_set, x_j_set_half;
  // per-thread scratch buffers (avoid heap allocation in hot paths)
  std::vector<std::vector<double>> A_mat_set;            // d*d per thread
  std::vector<std::vector<double>> basis_vals_set, basis_vals_half_set;             // cached collocation bases per thread
  std::vector<std::vector<double>> basis_prime_vals_set, basis_prime_vals_half_set; // cached collocation derivatives per thread
  std::vector<std::vector<double>> wA_set, wB_set;       // d per thread
  size_t dct_fft_size = 0, dct_fft_size_half = 0;
  std::vector<std::complex<double>> dct_chirp, dct_kernel_fft;
  std::vector<std::complex<double>> dct_chirp_half, dct_kernel_fft_half;
  std::map<BandedCacheKey, std::shared_ptr<const BandedSystemCache>> banded_system_cache;
  std::mutex banded_system_cache_mutex;
  size_t banded_system_cache_bytes = 0;
  std::map<BandedCacheKey, std::shared_ptr<DenseFallbackCache>> dense_fallback_cache;
  std::mutex dense_fallback_cache_mutex;
  size_t dense_fallback_cache_bytes = 0;
  uint64_t dense_fallback_cache_clock = 0;

  bool is_diagonal = false;
  bool is_x_log = false;
  bool speak_to_me = false;
  uint d, n_integrand, N_thread_max;
  bool system_of_equations_set = false;
  bool bisection_set = false;
  uint size_variables = 0;
  uint n_col = 8;
  uint type = 0;
  bool super_accurate = false;
  bool use_banded_levin = false;

  double min_interval = 0;
  double tol_rel = 1e-6;
  double tol_abs = 0.0;
  uint maximum_number_subintervals = 32;
  // Ratio cap: subintervals with b > max_ratio * max(a, x_t), x_t = min_i (ell_i + 1) / k_i,
  // are never treated as converged (0 disables). Prevents false convergence on
  // intervals spanning many decades in x.
  double max_ratio = 5.0;
  // Pieces with (sum_i k_i)(b - a) < non_oscillatory use interpolatory quadrature of
  // f w_0 on the collocation nodes instead of Levin (0 disables): there Levin's
  // collocation matrix is nearly singular.
  double non_oscillatory = 0.5;
  std::vector<double> smooth_q, smooth_q_half;
  // Quadrature weights of every stored piece, so that repeat calls only evaluate
  // sum_j weight_j f(x_j) on the collocation nodes; indexed like bisection.
  std::vector<std::vector<std::vector<double>>> weights_precomp;
  // Cubic spline coefficients (y_s, b_s, c_s, d_s) of each grid segment, y = y_s + t (b_s + t (c_s + d_s t))
  // with t = x - x_s, and f = y or f = exp(y) if is_y_log.
  std::vector<std::vector<std::array<double, 4>>> spline_coefficients;
  // sum_n weight_n t_n^p (p = 0..3) over the nodes of each stored integral in every grid segment,
  // so that repeat calls with linear y are sum_s moment_s . coefficients_s; indexed like bisection.
  std::vector<std::vector<SplineMoment>> spline_moments;
  // The nodes of each stored integral on the grid, so that repeat calls with log y are
  // sum_n weight_n exp(y(t_n)) without locating the nodes again; indexed like bisection.
  struct GridNode
  {
    uint segment;
    double t;
    double weight;
  };
  std::vector<std::vector<GridNode>> grid_nodes;
  // Samples of each integrand on the grid (log y if is_y_log).
  std::vector<std::vector<double>> spline_samples;
  // For linear y, each stored integral as weights on the samples, sum_a weight_a y_{start + a}:
  // the moments folded with the spline, used instead of them when cheaper. Then the spline
  // coefficients of linear-y integrands are only computed when needed (spline_coefficients_stale).
  struct SampleWeights
  {
    uint start = 0;
    std::vector<double> weight;
  };
  std::vector<SampleWeights> sample_weights;
  bool spline_coefficients_stale = false;

  void fill_w_double(double x, double k_1, double k_2, uint ell_1, uint ell_2, double *w);
  void fill_w_triple(double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, double *w);
  bool solve_LSE_single_banded(double A, double B, uint col, uint i_integrand, double k, uint ell);
  uint precomp_index(uint i_integrand, uint tid) const;
  bool is_non_oscillatory(double A, double B, double k_sum) const;
  double smooth_integral(double A, double B, uint col, uint i_integrand, const std::function<double(double)> &w0, bool store);
  void store_levin_weights(double A, double B, uint i_integrand, const double *wA, const double *wB);
  void factor_spline_grid();
  void set_spline_coefficients(uint i_integrand, const std::vector<double> &y, std::vector<double> &m, std::vector<double> &slope);
  void build_grid_tables();
  void fold_spline_moments(const std::vector<SplineMoment> &moments, SampleWeights &row, std::vector<double> &g_y,
                           std::vector<double> &g_s, std::vector<double> &g_m) const;
  void ensure_spline_coefficients();
  uint segment_of(double x, uint guess) const;
  std::vector<uint> integrand_indices(const std::optional<array_int> &integrand) const;
  double iterate_pieces(double A, double B, uint col, uint i_integrand, uint smax, bool verbose, double x_turn,
                        const std::function<double(double, double, uint)> &integrate, const std::string &label);
  void ensure_dense_single_scratch(uint tid, uint col);

public:
  pylevin(uint type_in, const array_double &x, const array_double &integrand, bool logx, bool logy, uint nthread, bool diagonal = false);

  ~pylevin();

  void init_splines(const array_double &x_in, const array_double &integrand_in, bool logx, bool logy);

  void set_levin(uint n_col_in, uint maximum_number_bisections_in, double relative_accuracy_in, bool super_accurate_in, bool verbose, double tol_abs_in, bool use_banded_levin_in = true, double max_ratio_in = 5.0, double non_oscillatory_in = 0.5);

  void update_integrand(const array_double &x, const array_double &integrand, bool logx, bool logy);

  array_double get_integrand(const array_double &x_in);

  pybind11::tuple get_quadrature();

  double w_single_bessel(double x, double k, uint ell, uint i);

  double w_double_bessel(double x, double k_1, double k_2, uint ell_1, uint ell_2, uint i);

  double w_triple_bessel(double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i);

  double A_matrix_single(uint i, uint j, double x, double k, uint ell);

  double A_matrix_double(uint i, uint j, double x, double k_1, double k_2, uint ell_1, uint ell_2);

  double A_matrix_triple(uint i, uint j, double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  void setNodes_cheby(uint i);

  double map_y_to_x(double y, double A, double B);

  double basis_function_cheby(double x, uint m);

  double basis_function_prime_cheby(double x, uint m);

  double inhomogeneity(double x, uint i_integrand, uint tid);

  void solve_LSE_single(double A, double B, uint col, uint i_integrand, double k, uint ell);

  void solve_LSE_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2);

  void solve_LSE_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  double p_cheby(double A, double B, uint i, double x, uint col, gsl_vector *c);

  double integrate_single(double A, double B, uint col, uint i_integrand, double k, uint ell);

  double integrate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2);

  double integrate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  double integrate_lse_set(uint i_integrand);

  double iterate_single(double A, double B, uint col, uint i_integrand, double k, uint ell, uint smax, bool verbose);

  double iterate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2, uint smax, bool verbose);

  double iterate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint smax, bool verbose);

  double levin_integrate_single_bessel(double x_min, double x_max, double k, uint ell, uint i_integrand);

  double levin_integrate_double_bessel(double x_min, double x_max, double k_1, double k_2, uint ell_1, uint ell_2, uint i_integrand);

  double levin_integrate_triple_bessel(double x_min, double x_max, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i_integrand);

  void allocate_variables_for_lse();

  void levin_integrate_bessel_single(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_in, const array_int &ell_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in = std::nullopt);

  void levin_integrate_bessel_double(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_1_in, const array_double &k_2_in, const array_int &ell_1_in, const array_int &ell_2_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in = std::nullopt);

  void levin_integrate_bessel_triple(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_1_in, const array_double &k_2_in, const array_double &k_3_in, const array_int &ell_1_in, const array_int &ell_2_in, const array_int &ell_3_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in = std::nullopt);
};

#endif

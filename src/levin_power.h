#ifndef LEVIN_POWER_H
#define LEVIN_POWER_H

#include <vector>
#include <numeric>
#include <omp.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>


#include <cmath>
#include <thread>

class levin_power
{
private:
    std::vector<gsl_interp_accel *> acc_integrand;
    std::vector<gsl_spline *> spline_integrand;
    std::vector<bool> is_y_log;
    std::vector<uint> index_variable, index_integral, index_bisection;
    std::vector<std::vector<std::vector<double>>> bisection;
    std::vector<std::vector<std::vector<gsl_matrix *>>> LU_G_matrix;
    std::vector<std::vector<std::vector<gsl_permutation *>>> permutation;
    std::vector<std::vector<std::vector<std::vector<double>>>> basis_precomp;
    std::vector<std::vector<std::vector<std::vector<double>>>> w_precomp;

    bool is_x_log = false;
    bool speak_to_me = false;
    uint d, n_integrand, N_thread_max;
    bool error_count = false;
    bool system_of_equations_set = false;
    bool bisection_set = false;
    uint size_variables = 0;
    uint n_col = 8;
    uint type = 0;
    bool super_accurate = false;
    
    double min_interval = 1e-5;
    double tol_rel = 1e-6;
    double tol_abs = 1e-40;
    uint maximum_number_subintervals = 32;

    gsl_error_handler_t *old_handler;

    void set_pointer();

public:
  
  levin_power(uint type_in, std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy, uint nthread);

  /**
   * Destructor: clean up all allocated memory.
   */
  ~levin_power();

  void init_splines(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy);

  void set_levin(uint n_col_in, uint maximum_number_bisections_in, double relative_accuracy_in, bool super_accurate_in, bool verbose);

  void update_integrand(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy);
  
  std::vector<std::vector<std::vector<double>>> get_bisection();
  
  std::vector<std::vector<double>> get_integrand(std::vector<double> x);

  double w_single_bessel(double x, double k, uint ell, uint i);

  double w_double_bessel(double x, double k_1, double k_2, uint ell_1, uint ell_2, uint i);

  double w_triple_bessel(double x, double k_1, double k_2, double k_3,  uint ell_1, uint ell_2, uint ell_3, uint i);

  double A_matrix_single(uint i, uint j, double x, double k, uint ell);

  double A_matrix_double(uint i, uint j, double x, double k_1, double k_2, uint ell_1, uint ell_2);

  double A_matrix_triple(uint i, uint j, double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  std::vector<double> setNodes(double A, double B, uint col);

  double basis_function(double A, double B, double x, uint m);

  double basis_function_prime(double A, double B, double x, uint m);

  double inhomogeneity(double x, uint i_integrand);

  std::vector<double> solve_LSE_single(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k, uint ell);

  std::vector<double> solve_LSE_double(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2);

  std::vector<double> solve_LSE_triple(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  double p(double A, double B, uint i, double x, uint col, std::vector<double> c);

  std::vector<double> p_precompute(double A, double B, uint i, double x, uint col, std::vector<double> c);

  double integrate_single(double A,  double B, uint col, uint i_integrand, double k, uint ell);

  double integrate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2);

  double integrate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  double integrate_lse_set_single(double A, double B, uint col, uint i_integrand, double k, uint ell);

  double integrate_lse_set_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2);

  double integrate_lse_set_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3);

  double iterate_single(double A, double B, uint col, uint i_integrand, double k, uint ell, uint smax, bool verbose);

  double iterate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2, uint smax, bool verbose);

  double iterate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint smax, bool verbose);

  double levin_integrate_single_bessel(double  x_min, double x_max, double k, uint ell, uint i_integrand);

  double levin_integrate_double_bessel(double  x_min, double x_max, double k_1, double k_2, uint ell_1, uint ell_2, uint i_integrand);

  double levin_integrate_triple_bessel(double  x_min, double x_max, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i_integrand);

  std::vector<std::vector<double>> levin_integrate_bessel_single(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k, std::vector<uint> ell, bool diagonal);

  std::vector<std::vector<double>> levin_integrate_bessel_double(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<uint> ell_1, std::vector<uint> ell_2, bool diagonal);

  std::vector<std::vector<double>> levin_integrate_bessel_triple(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<double> k_3, std::vector<uint> ell_1, std::vector<uint> ell_2, std::vector<uint> ell_3, bool diagonal);

};

#endif
#include "pylevin.h"
#include <gsl/gsl_spline.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/chebyshev.hpp>

pylevin::pylevin(uint type_in, std::vector<double> x, const std::vector<std::vector<double>> &integrand, bool logx, bool logy, uint nthread, bool diagonal)
{
    is_diagonal = diagonal;
    if (integrand.size() != x.size())
    {
        throw std::range_error("support dimensions must match integrand dimensions");
    }
    type = type_in;
    if (type < 2)
    {
        d = 2;
    }
    else
    {
        if (type < 4)
        {
            d = 4;
        }
        else
        {
            d = 8;
        }
    }
    N_thread_max = nthread;
    init_splines(x, integrand, logx, logy);
    index_variable.resize(N_thread_max);
    index_integral.resize(N_thread_max);
    index_bisection.resize(N_thread_max);
    set_levin(8, 32, 1e-4, false, false, 0.0);
}

pylevin::~pylevin()
{
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
    for (uint i = 0; i < n_integrand; i++)
    {
        gsl_spline_free(spline_integrand[i]);
        for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
        {
            gsl_interp_accel_free(acc_integrand[i][i_thread]);
        }
    }
    if (system_of_equations_set)
    {
        if (is_diagonal)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                for (uint i_bisec = 0; i_bisec < bisection[i_integrand].size() - 1; i_bisec++)
                {
                    gsl_matrix_free(LU_G_matrix[i_integrand][i_bisec]);
                    gsl_permutation_free(permutation[i_integrand][i_bisec]);
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                    {
                        gsl_matrix_free(LU_G_matrix[i_integrand * size_variables + i_variable][i_bisec]);
                        gsl_permutation_free(permutation[i_integrand * size_variables + i_variable][i_bisec]);
                    }
                }
            }
        }
    }
    for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
    {
        gsl_vector_free(F_stacked_set[i_thread]);
        gsl_vector_free(F_stacked_set_half[i_thread]);
        gsl_vector_free(ce_set[i_thread]);
        gsl_vector_free(ce_set_half[i_thread]);
        gsl_matrix_free(matrix_G_set[i_thread]);
        gsl_matrix_free(matrix_G_set_half[i_thread]);
        gsl_permutation_free(P_set[i_thread]);
        gsl_permutation_free(P_set_half[i_thread]);
    }
}

void pylevin::set_levin(uint n_col_in, uint maximum_number_bisections_in, double relative_accuracy_in, bool super_accurate_in, bool verbose, double tol_abs_in)
{
    tol_abs = tol_abs_in;
    n_col = (n_col_in + 1) / 2;
    n_col *= 2;
    maximum_number_subintervals = maximum_number_bisections_in;
    tol_rel = relative_accuracy_in;
    speak_to_me = verbose;
    super_accurate = super_accurate_in;
    // Free previously allocated per-thread GSL objects to avoid leaks on re-call
    if (!matrix_G_set.empty())
    {
        for (uint i_thread = 0; i_thread < matrix_G_set.size(); i_thread++)
        {
            gsl_vector_free(F_stacked_set[i_thread]);
            gsl_vector_free(F_stacked_set_half[i_thread]);
            gsl_vector_free(ce_set[i_thread]);
            gsl_vector_free(ce_set_half[i_thread]);
            gsl_matrix_free(matrix_G_set[i_thread]);
            gsl_matrix_free(matrix_G_set_half[i_thread]);
            gsl_permutation_free(P_set[i_thread]);
            gsl_permutation_free(P_set_half[i_thread]);
        }
    }
    F_stacked_set.resize(N_thread_max);
    F_stacked_set_half.resize(N_thread_max);
    ce_set.resize(N_thread_max);
    ce_set_half.resize(N_thread_max);
    matrix_G_set.resize(N_thread_max);
    matrix_G_set_half.resize(N_thread_max);
    P_set.resize(N_thread_max);
    P_set_half.resize(N_thread_max);
    if (!x_j_set.empty())
    {
        x_j_set.clear();
        x_j_set_half.clear();
    }
    x_j_set.resize(N_thread_max, std::vector<double>(n_col));
    x_j_set_half.resize(N_thread_max, std::vector<double>(n_col / 2));
    A_mat_set.assign(N_thread_max, std::vector<double>(d * d));
    basis_vals_set.assign(N_thread_max, std::vector<double>(n_col));
    basis_prime_vals_set.assign(N_thread_max, std::vector<double>(n_col));
    wA_set.assign(N_thread_max, std::vector<double>(d));
    wB_set.assign(N_thread_max, std::vector<double>(d));
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
    for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
    {
        F_stacked_set[i_thread] = gsl_vector_alloc(d * n_col);
        F_stacked_set_half[i_thread] = gsl_vector_alloc(d * n_col / 2);
        ce_set[i_thread] = gsl_vector_alloc(d * n_col);
        ce_set_half[i_thread] = gsl_vector_alloc(d * n_col / 2);
        matrix_G_set[i_thread] = gsl_matrix_alloc(d * n_col, d * n_col);
        matrix_G_set_half[i_thread] = gsl_matrix_alloc(d * n_col / 2, d * n_col / 2);
        P_set[i_thread] = gsl_permutation_alloc(d * n_col);
        P_set_half[i_thread] = gsl_permutation_alloc(d * n_col / 2);
        gsl_vector_set_zero(F_stacked_set[i_thread]);
        gsl_vector_set_zero(F_stacked_set_half[i_thread]);
        setNodes_cheby(i_thread);
    }
}

void pylevin::init_splines(std::vector<double> &x, const std::vector<std::vector<double>> &integrand, bool logx, bool logy)
{
    n_integrand = integrand[0].size();
    if (!system_of_equations_set && !bisection_set)
    {
        spline_integrand.resize(integrand[0].size());
        acc_integrand.resize(integrand[0].size(), std::vector<gsl_interp_accel *>(N_thread_max));
        is_y_log.resize(n_integrand, false);
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            spline_integrand[i_integrand] = gsl_spline_alloc(gsl_interp_akima, x.size());
            for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
            {
                acc_integrand[i_integrand][i_thread] = gsl_interp_accel_alloc();
            }
        }
    }
    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
    {
        if (spline_integrand.size() != n_integrand)
        {
            std::cout << spline_integrand.size() << " " << n_integrand << std::endl;
            throw std::range_error("If you update the integrand, they have to have the same shapes as in the constructor");
        }
        std::vector<double> init_weight(x.size(), 0.0);
        if (logy)
        {
            is_y_log[i_integrand] = true;
            for (uint i = 0; i < x.size(); i++)
            {
                if (integrand[i][i_integrand] <= 0)
                {
                    is_y_log[i_integrand] = false;
                }
            }
        }
        else
        {
            is_y_log[i_integrand] = false;
        }
        for (uint i = 0; i < x.size(); i++)
        {
            if (i_integrand == 0)
            {
                if (logx)
                {
                    is_x_log = true;
                    x[i] = log(x[i]);
                }
            }
            if (is_y_log[i_integrand])
            {
                init_weight[i] = log(integrand[i][i_integrand]);
            }
            else
            {
                init_weight[i] = integrand[i][i_integrand];
            }
        }
        gsl_spline_init(spline_integrand[i_integrand], &x[0], &init_weight[0], x.size());
    }
}

void pylevin::update_integrand(std::vector<double> x, const std::vector<std::vector<double>> &integrand, bool logx, bool logy)
{
    init_splines(x, integrand, logx, logy);
}

std::vector<std::vector<double>> pylevin::get_integrand(std::vector<double> x)
{
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(n_integrand));
    for (uint i_x = 0; i_x < x.size(); i_x++)
    {
        double x_value = x[i_x];
        if (is_x_log)
        {
            x_value = log(x_value);
        }
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            result[i_x][i_integrand] = gsl_spline_eval(spline_integrand[i_integrand], x_value, acc_integrand[i_integrand][0]);
            if (is_y_log[i_integrand])
            {
                result[i_x][i_integrand] = exp(result[i_x][i_integrand]);
            }
        }
    }
    return result;
}

double pylevin::w_single_bessel(double x, double k, uint ell, uint i)
{
    if (super_accurate == false)
    {
        if (type == 0)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_jl(ell, x * k);
            case 1:
                return gsl_sf_bessel_jl(ell + 1, x * k);
            default:
                return 0.0;
            }
        }
        if (type == 1)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_Jn(ell, x * k);
            case 1:
                return gsl_sf_bessel_Jn(ell + 1, x * k);
            default:
                return 0.0;
            }
        }
    }
    else
    {
        if (type == 0)
        {
            switch (i)
            {
            case 0:
                return boost::math::sph_bessel(ell, x * k);
            case 1:
                return boost::math::sph_bessel(ell + 1, x * k);
            default:
                return 0.0;
            }
        }
        if (type == 1)
        {
            switch (i)
            {
            case 0:
                return boost::math::cyl_bessel_j(ell, x * k);
            case 1:
                return boost::math::cyl_bessel_j(ell + 1, x * k);
            default:
                return 0.0;
            }
        }
    }
    return 0.0;
}

double pylevin::w_double_bessel(double x, double k_1, double k_2, uint ell_1, uint ell_2, uint i)
{
    if (super_accurate == false)
    {
        if (type == 2)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_1, x * k_1);
            case 1:
                return gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_1 + 1, x * k_1);
            case 2:
                return gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_1, x * k_1);
            case 3:
                return gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_1 + 1, x * k_1);
            }
        }
        if (type == 3)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_1, x * k_1);
            case 1:
                return gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_1 + 1, x * k_1);
            case 2:
                return gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_1, x * k_1);
            case 3:
                return gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_1 + 1, x * k_1);
            }
        }
    }
    else
    {
        if (type == 2)
        {
            switch (i)
            {
            case 0:
                return boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_1, x * k_1);
                break;
            case 1:
                return boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_1 + 1, x * k_1);
                break;
            case 2:
                return boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_1, x * k_1);
                break;
            case 3:
                return boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_1 + 1, x * k_1);
                break;
            }
        }
        if (type == 3)
        {
            switch (i)
            {
            case 0:
                return boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_1, x * k_1);
                break;
            case 1:
                return boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_1 + 1, x * k_1);
                break;
            case 2:
                return boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_1, x * k_1);
                break;
            case 3:
                return boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_1 + 1, x * k_1);
                break;
            }
        }
    }
    return 0.0;
}

double pylevin::w_triple_bessel(double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i)
{
    if (super_accurate == false)
    {
        if (type == 4)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_jl(ell_1, x * k_1) * gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_3, x * k_3);
            case 1:
                return gsl_sf_bessel_jl(ell_1 + 1, x * k_1) * gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_3, x * k_3);
            case 2:
                return gsl_sf_bessel_jl(ell_1, x * k_1) * gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_3, x * k_3);
            case 3:
                return gsl_sf_bessel_jl(ell_1, x * k_1) * gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_3 + 1, x * k_3);
            case 4:
                return gsl_sf_bessel_jl(ell_1 + 1, x * k_1) * gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_3, x * k_3);
            case 5:
                return gsl_sf_bessel_jl(ell_1, x * k_1) * gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_3 + 1, x * k_3);
            case 6:
                return gsl_sf_bessel_jl(ell_1 + 1, x * k_1) * gsl_sf_bessel_jl(ell_2, x * k_2) * gsl_sf_bessel_jl(ell_3 + 1, x * k_3);
            case 7:
                return gsl_sf_bessel_jl(ell_1 + 1, x * k_1) * gsl_sf_bessel_jl(ell_2 + 1, x * k_2) * gsl_sf_bessel_jl(ell_3 + 1, x * k_3);
            }
        }
        if (type == 5)
        {
            switch (i)
            {
            case 0:
                return gsl_sf_bessel_Jn(ell_1, x * k_1) * gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_3, x * k_3);
            case 1:
                return gsl_sf_bessel_Jn(ell_1 + 1, x * k_1) * gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_3, x * k_3);
            case 2:
                return gsl_sf_bessel_Jn(ell_1, x * k_1) * gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_3, x * k_3);
            case 3:
                return gsl_sf_bessel_Jn(ell_1, x * k_1) * gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_3 + 1, x * k_3);
            case 4:
                return gsl_sf_bessel_Jn(ell_1 + 1, x * k_1) * gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_3, x * k_3);
            case 5:
                return gsl_sf_bessel_Jn(ell_1, x * k_1) * gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_3 + 1, x * k_3);
            case 6:
                return gsl_sf_bessel_Jn(ell_1 + 1, x * k_1) * gsl_sf_bessel_Jn(ell_2, x * k_2) * gsl_sf_bessel_Jn(ell_3 + 1, x * k_3);
            case 7:
                return gsl_sf_bessel_Jn(ell_1 + 1, x * k_1) * gsl_sf_bessel_Jn(ell_2 + 1, x * k_2) * gsl_sf_bessel_Jn(ell_3 + 1, x * k_3);
            }
        }
    }
    else
    {
        if (type == 4)
        {
            switch (i)
            {
            case 0:
                return boost::math::sph_bessel(ell_1, x * k_1) * boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_3, x * k_3);
            case 1:
                return boost::math::sph_bessel(ell_1 + 1, x * k_1) * boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_3, x * k_3);
            case 2:
                return boost::math::sph_bessel(ell_1, x * k_1) * boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_3, x * k_3);
            case 3:
                return boost::math::sph_bessel(ell_1, x * k_1) * boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_3 + 1, x * k_3);
            case 4:
                return boost::math::sph_bessel(ell_1 + 1, x * k_1) * boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_3, x * k_3);
            case 5:
                return boost::math::sph_bessel(ell_1, x * k_1) * boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_3 + 1, x * k_3);
            case 6:
                return boost::math::sph_bessel(ell_1 + 1, x * k_1) * boost::math::sph_bessel(ell_2, x * k_2) * boost::math::sph_bessel(ell_3 + 1, x * k_3);
            case 7:
                return boost::math::sph_bessel(ell_1 + 1, x * k_1) * boost::math::sph_bessel(ell_2 + 1, x * k_2) * boost::math::sph_bessel(ell_3 + 1, x * k_3);
            }
        }
        if (type == 5)
        {
            switch (i)
            {
            case 0:
                return boost::math::cyl_bessel_j(ell_1, x * k_1) * boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_3, x * k_3);
            case 1:
                return boost::math::cyl_bessel_j(ell_1 + 1, x * k_1) * boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_3, x * k_3);
            case 2:
                return boost::math::cyl_bessel_j(ell_1, x * k_1) * boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_3, x * k_3);
            case 3:
                return boost::math::cyl_bessel_j(ell_1, x * k_1) * boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_3 + 1, x * k_3);
            case 4:
                return boost::math::cyl_bessel_j(ell_1 + 1, x * k_1) * boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_3, x * k_3);
            case 5:
                return boost::math::cyl_bessel_j(ell_1, x * k_1) * boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_3 + 1, x * k_3);
            case 6:
                return boost::math::cyl_bessel_j(ell_1 + 1, x * k_1) * boost::math::cyl_bessel_j(ell_2, x * k_2) * boost::math::cyl_bessel_j(ell_3 + 1, x * k_3);
            case 7:
                return boost::math::cyl_bessel_j(ell_1 + 1, x * k_1) * boost::math::cyl_bessel_j(ell_2 + 1, x * k_2) * boost::math::cyl_bessel_j(ell_3 + 1, x * k_3);
            }
        }
    }
    return 0.0;
}

void pylevin::fill_w_double(double x, double k_1, double k_2, uint ell_1, uint ell_2, double *w)
{
    if (!super_accurate)
    {
        if (type == 2)
        {
            double b1a = gsl_sf_bessel_jl(ell_1, x * k_1);
            double b1b = gsl_sf_bessel_jl(ell_1 + 1, x * k_1);
            double b2a = gsl_sf_bessel_jl(ell_2, x * k_2);
            double b2b = gsl_sf_bessel_jl(ell_2 + 1, x * k_2);
            w[0] = b1a * b2a;
            w[1] = b1b * b2a;
            w[2] = b1a * b2b;
            w[3] = b1b * b2b;
        }
        else if (type == 3)
        {
            double b1a = gsl_sf_bessel_Jn(ell_1, x * k_1);
            double b1b = gsl_sf_bessel_Jn(ell_1 + 1, x * k_1);
            double b2a = gsl_sf_bessel_Jn(ell_2, x * k_2);
            double b2b = gsl_sf_bessel_Jn(ell_2 + 1, x * k_2);
            w[0] = b1a * b2a;
            w[1] = b1b * b2a;
            w[2] = b1a * b2b;
            w[3] = b1b * b2b;
        }
    }
    else
    {
        if (type == 2)
        {
            double b1a = boost::math::sph_bessel(ell_1, x * k_1);
            double b1b = boost::math::sph_bessel(ell_1 + 1, x * k_1);
            double b2a = boost::math::sph_bessel(ell_2, x * k_2);
            double b2b = boost::math::sph_bessel(ell_2 + 1, x * k_2);
            w[0] = b1a * b2a;
            w[1] = b1b * b2a;
            w[2] = b1a * b2b;
            w[3] = b1b * b2b;
        }
        else if (type == 3)
        {
            double b1a = boost::math::cyl_bessel_j(ell_1, x * k_1);
            double b1b = boost::math::cyl_bessel_j(ell_1 + 1, x * k_1);
            double b2a = boost::math::cyl_bessel_j(ell_2, x * k_2);
            double b2b = boost::math::cyl_bessel_j(ell_2 + 1, x * k_2);
            w[0] = b1a * b2a;
            w[1] = b1b * b2a;
            w[2] = b1a * b2b;
            w[3] = b1b * b2b;
        }
    }
}

void pylevin::fill_w_triple(double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, double *w)
{
    if (!super_accurate)
    {
        if (type == 4)
        {
            double b1a = gsl_sf_bessel_jl(ell_1, x * k_1);
            double b1b = gsl_sf_bessel_jl(ell_1 + 1, x * k_1);
            double b2a = gsl_sf_bessel_jl(ell_2, x * k_2);
            double b2b = gsl_sf_bessel_jl(ell_2 + 1, x * k_2);
            double b3a = gsl_sf_bessel_jl(ell_3, x * k_3);
            double b3b = gsl_sf_bessel_jl(ell_3 + 1, x * k_3);
            w[0] = b1a * b2a * b3a;
            w[1] = b1b * b2a * b3a;
            w[2] = b1a * b2b * b3a;
            w[3] = b1a * b2a * b3b;
            w[4] = b1b * b2b * b3a;
            w[5] = b1a * b2b * b3b;
            w[6] = b1b * b2a * b3b;
            w[7] = b1b * b2b * b3b;
        }
        else if (type == 5)
        {
            double b1a = gsl_sf_bessel_Jn(ell_1, x * k_1);
            double b1b = gsl_sf_bessel_Jn(ell_1 + 1, x * k_1);
            double b2a = gsl_sf_bessel_Jn(ell_2, x * k_2);
            double b2b = gsl_sf_bessel_Jn(ell_2 + 1, x * k_2);
            double b3a = gsl_sf_bessel_Jn(ell_3, x * k_3);
            double b3b = gsl_sf_bessel_Jn(ell_3 + 1, x * k_3);
            w[0] = b1a * b2a * b3a;
            w[1] = b1b * b2a * b3a;
            w[2] = b1a * b2b * b3a;
            w[3] = b1a * b2a * b3b;
            w[4] = b1b * b2b * b3a;
            w[5] = b1a * b2b * b3b;
            w[6] = b1b * b2a * b3b;
            w[7] = b1b * b2b * b3b;
        }
    }
    else
    {
        if (type == 4)
        {
            double b1a = boost::math::sph_bessel(ell_1, x * k_1);
            double b1b = boost::math::sph_bessel(ell_1 + 1, x * k_1);
            double b2a = boost::math::sph_bessel(ell_2, x * k_2);
            double b2b = boost::math::sph_bessel(ell_2 + 1, x * k_2);
            double b3a = boost::math::sph_bessel(ell_3, x * k_3);
            double b3b = boost::math::sph_bessel(ell_3 + 1, x * k_3);
            w[0] = b1a * b2a * b3a;
            w[1] = b1b * b2a * b3a;
            w[2] = b1a * b2b * b3a;
            w[3] = b1a * b2a * b3b;
            w[4] = b1b * b2b * b3a;
            w[5] = b1a * b2b * b3b;
            w[6] = b1b * b2a * b3b;
            w[7] = b1b * b2b * b3b;
        }
        else if (type == 5)
        {
            double b1a = boost::math::cyl_bessel_j(ell_1, x * k_1);
            double b1b = boost::math::cyl_bessel_j(ell_1 + 1, x * k_1);
            double b2a = boost::math::cyl_bessel_j(ell_2, x * k_2);
            double b2b = boost::math::cyl_bessel_j(ell_2 + 1, x * k_2);
            double b3a = boost::math::cyl_bessel_j(ell_3, x * k_3);
            double b3b = boost::math::cyl_bessel_j(ell_3 + 1, x * k_3);
            w[0] = b1a * b2a * b3a;
            w[1] = b1b * b2a * b3a;
            w[2] = b1a * b2b * b3a;
            w[3] = b1a * b2a * b3b;
            w[4] = b1b * b2b * b3a;
            w[5] = b1a * b2b * b3b;
            w[6] = b1b * b2a * b3b;
            w[7] = b1b * b2b * b3b;
        }
    }
}

double pylevin::A_matrix_single(uint i, uint j, double x, double k, uint ell)
{
    if (type == 0)
    {
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell) / x;
        }
        if (i * j == 1)
        {
            return -(ell + 2.0) / x;
        }
        if (i < j)
        {
            return -k;
        }
        else
        {
            return k;
        }
    }
    if (type == 1)
    {
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell) / x;
        }
        if (i * j == 1)
        {
            return -(ell + 1.0) / x;
        }
        if (i < j)
        {
            return -k;
        }
        else
        {
            return k;
        }
    }
    return 0.0;
}

double pylevin::A_matrix_double(uint i, uint j, double x, double k_1, double k_2, uint ell_1, uint ell_2)
{
    if (type == 2)
    {
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return (ell_1 + ell_2) / x;
        }
        if (i == 1 && j == 1)
        {
            return -(static_cast<double>(ell_1) - static_cast<double>(ell_2) + 2.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2) - 2.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return -(static_cast<double>(ell_1) + static_cast<double>(ell_2) + 4.0) / x;
        }
        if ((i == 1 && j == 0) || (i == 3 && j == 2))
        {
            return k_1;
        }
        if ((i == 2 && j == 0) || (i == 3 && j == 1))
        {
            return k_2;
        }
        if ((i == 0 && j == 1) || (i == 2 && j == 3))
        {
            return -k_1;
        }
        if ((i == 0 && j == 2) || (i == 1 && j == 3))
        {
            return -k_2;
        }
    }
    if (type == 3)
    {
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell_1 + ell_2) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1) - 1.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2) - 1.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return -(ell_1 + ell_2 + 2.0) / x;
        }
        if ((i == 1 && j == 0) || (i == 3 && j == 2))
        {
            return k_1;
        }
        if ((i == 2 && j == 0) || (i == 3 && j == 1))
        {
            return k_2;
        }
        if ((i == 0 && j == 1) || (i == 2 && j == 3))
        {
            return -k_1;
        }
        if ((i == 0 && j == 2) || (i == 1 && j == 3))
        {
            return -k_2;
        }
    }
    return 0.0;
}

double pylevin::A_matrix_triple(uint i, uint j, double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    if (type == 4)
    {
        if (i == 0 && j == 1)
        {
            return -k_1;
        }
        if (i == 0 && j == 2)
        {
            return -k_2;
        }
        if (i == 0 && j == 3)
        {
            return -k_3;
        }
        if (i == 1 && j == 0)
        {
            return k_1;
        }
        if (j == 0 && i == 2)
        {
            return k_2;
        }
        if (j == 0 && i == 3)
        {
            return k_3;
        }
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell_1 + ell_2 + ell_3) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_2 + ell_3) - static_cast<double>(ell_1) - 2.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1 + ell_3) - static_cast<double>(ell_2) - 2.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return (static_cast<double>(ell_1 + ell_2) - static_cast<double>(ell_3) - 2.0) / x;
        }
        if (i == 1 && j == 4)
        {
            return -k_2;
        }
        if (j == 1 && i == 4)
        {
            return k_2;
        }
        if (i == 1 && j == 6)
        {
            return -k_3;
        }
        if (j == 1 && i == 6)
        {
            return k_3;
        }
        if (i == 2 && j == 4)
        {
            return -k_1;
        }
        if (j == 2 && i == 4)
        {
            return k_1;
        }
        if (i == 2 && j == 5)
        {
            return -k_3;
        }
        if (j == 2 && i == 5)
        {
            return k_3;
        }
        if (i == 3 && j == 5)
        {
            return -k_2;
        }
        if (j == 3 && i == 5)
        {
            return k_2;
        }
        if (i == 3 && j == 6)
        {
            return -k_1;
        }
        if (j == 3 && i == 6)
        {
            return k_1;
        }
        if (i == 4 && j == 7)
        {
            return -k_3;
        }
        if (j == 4 && i == 7)
        {
            return k_3;
        }
        if (i == 5 && j == 7)
        {
            return -k_1;
        }
        if (j == 5 && i == 7)
        {
            return k_1;
        }
        if (i == 6 && j == 7)
        {
            return -k_2;
        }
        if (j == 6 && i == 7)
        {
            return k_2;
        }
        if (i == 4 && j == 4)
        {
            return (static_cast<double>(ell_3) - static_cast<double>(ell_1 + ell_2) - 4.0) / x;
        }
        if (i == 5 && j == 5)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2 + ell_3) - 4.0) / x;
        }
        if (i == 6 && j == 6)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1 + ell_3) - 4.0) / x;
        }
        if (i == 7 && j == 7)
        {
            return (-static_cast<double>(ell_2 + ell_3 + ell_1) - 6.0) / x;
        }
        return 0.0;
    }
    if (type == 5)
    {
        if (i == 0 && j == 1)
        {
            return -k_1;
        }
        if (i == 0 && j == 2)
        {
            return -k_2;
        }
        if (i == 0 && j == 3)
        {
            return -k_3;
        }
        if (i == 1 && j == 0)
        {
            return k_1;
        }
        if (j == 0 && i == 2)
        {
            return k_2;
        }
        if (j == 0 && i == 3)
        {
            return k_3;
        }
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell_1 + ell_2 + ell_3) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_2 + ell_3) - static_cast<double>(ell_1) - 1.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1 + ell_3) - static_cast<double>(ell_2) - 1.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return (static_cast<double>(ell_1 + ell_2) - static_cast<double>(ell_3) - 1.0) / x;
        }
        if (i == 1 && j == 4)
        {
            return -k_2;
        }
        if (j == 1 && i == 4)
        {
            return k_2;
        }
        if (i == 1 && j == 6)
        {
            return -k_3;
        }
        if (j == 1 && i == 6)
        {
            return k_3;
        }
        if (i == 2 && j == 4)
        {
            return -k_1;
        }
        if (j == 2 && i == 4)
        {
            return k_1;
        }
        if (i == 2 && j == 5)
        {
            return -k_3;
        }
        if (j == 2 && i == 5)
        {
            return k_3;
        }
        if (i == 3 && j == 5)
        {
            return -k_2;
        }
        if (j == 3 && i == 5)
        {
            return k_2;
        }
        if (i == 3 && j == 6)
        {
            return -k_1;
        }
        if (j == 3 && i == 6)
        {
            return k_1;
        }
        if (i == 4 && j == 7)
        {
            return -k_3;
        }
        if (j == 4 && i == 7)
        {
            return k_3;
        }
        if (i == 5 && j == 7)
        {
            return -k_1;
        }
        if (j == 5 && i == 7)
        {
            return k_1;
        }
        if (i == 6 && j == 7)
        {
            return -k_2;
        }
        if (j == 6 && i == 7)
        {
            return k_2;
        }
        if (i == 4 && j == 4)
        {
            return (static_cast<double>(ell_3) - static_cast<double>(ell_1 + ell_2) - 2.0) / x;
        }
        if (i == 5 && j == 5)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2 + ell_3) - 2.0) / x;
        }
        if (i == 6 && j == 6)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1 + ell_3) - 2.0) / x;
        }
        if (i == 7 && j == 7)
        {
            return (-static_cast<double>(ell_2 + ell_3 + ell_1) - 3.0) / x;
        }
        return 0.0;
    }
    return 0.0;
}

void pylevin::setNodes_cheby(uint i)
{
    const double norm = -1.0 / cos(M_PI * ((1. + 2 * n_col) / (2 * n_col)));
    for (uint j = 0; j < n_col; j++)
    {
        x_j_set[i][j] = norm * cos((2. * (j + 1) - 1) / (2. * n_col) * M_PI + M_PI);
    }
    const double norm_half = -1.0 / cos(M_PI * ((1. + n_col) / n_col));
    for (uint j = 0; j < n_col / 2; j++)
    {
        x_j_set_half[i][j] = norm_half * cos((2. * (j + 1) - 1) / (n_col)*M_PI + M_PI);
    }
}

double pylevin::map_y_to_x(double y, double A, double B)
{
    return -(A * (y - 1.) - B * (y + 1.)) / 2;
}

double pylevin::basis_function_cheby(double x, uint m)
{
    return boost::math::chebyshev_t(m, x);
}

double pylevin::basis_function_prime_cheby(double x, uint m)
{
    return boost::math::chebyshev_t_prime(m, x);
}

double pylevin::inhomogeneity(double x, uint i_integrand, uint tid)
{
    if (is_x_log)
    {
        x = log(x);
    }
    double result;
    int status;
    status = gsl_spline_eval_e(spline_integrand[i_integrand], x, acc_integrand[i_integrand][tid], &result);
    if (status)
    {
        return 0;
    }
    if (is_y_log[i_integrand])
    {
        result = exp(result);
    }
    return result;
}

void pylevin::solve_LSE_single(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    uint tid = omp_get_thread_num();
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_matrix_set_zero(matrix_G);
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    std::vector<double> &basis_vals = basis_vals_set[tid];
    std::vector<double> &basis_prime_vals = basis_prime_vals_set[tid];
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        for (uint m = 0; m < col; m++)
        {
            basis_vals[m] = basis_function_cheby(y, m);
            basis_prime_vals[m] = basis_function_prime_cheby(y, m);
        }
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_single(q, i, x, k, ell);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];

    if (bisection_set && !system_of_equations_set)
    {
        if (!is_diagonal)
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]]);
        }
        else
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand][index_bisection[tid]]);
        }
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
    else
    {
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
}

void pylevin::solve_LSE_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    uint tid = omp_get_thread_num();
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_matrix_set_zero(matrix_G);
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    std::vector<double> &basis_vals = basis_vals_set[tid];
    std::vector<double> &basis_prime_vals = basis_prime_vals_set[tid];
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        for (uint m = 0; m < col; m++)
        {
            basis_vals[m] = basis_function_cheby(y, m);
            basis_prime_vals[m] = basis_function_prime_cheby(y, m);
        }
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_double(q, i, x, k_1, k_2, ell_1, ell_2);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];

    if (bisection_set && !system_of_equations_set)
    {
        if (!is_diagonal)
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]]);
        }
        else
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand][index_bisection[tid]]);
        }
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
    else
    {
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
}

void pylevin::solve_LSE_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    uint tid = omp_get_thread_num();
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_matrix_set_zero(matrix_G);
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    std::vector<double> &basis_vals = basis_vals_set[tid];
    std::vector<double> &basis_prime_vals = basis_prime_vals_set[tid];
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        for (uint m = 0; m < col; m++)
        {
            basis_vals[m] = basis_function_cheby(y, m);
            basis_prime_vals[m] = basis_function_prime_cheby(y, m);
        }
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_triple(q, i, x, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];

    if (bisection_set && !system_of_equations_set)
    {
        if (!is_diagonal)
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]]);
        }
        else
        {
            gsl_linalg_LU_decomp(matrix_G, permutation[i_integrand][index_bisection[tid]], &s);
            gsl_matrix_memcpy(LU_G_matrix[i_integrand][index_bisection[tid]], matrix_G);
            gsl_permutation_memcpy(P, permutation[i_integrand][index_bisection[tid]]);
        }
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
    else
    {
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
    }
}

double pylevin::p_cheby(double A, double B, uint i, double x, uint col, gsl_vector *c)
{
    double result = 0.0;
    for (uint m = 0; m < col; m++)
    {
        result += gsl_vector_get(c, i * col + m) * basis_function_cheby(x, m);
    }
    return result;
}

double pylevin::integrate_single(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    if (A == B)
    {
        return 0.;
    }
    uint tid = omp_get_thread_num();
    double result = 0.0;
    solve_LSE_single(A, B, col, i_integrand, k, ell);
    double *wA = wA_set[tid].data();
    double *wB = wB_set[tid].data();
    for (uint i = 0; i < d; i++)
    {
        wA[i] = w_single_bessel(A, k, ell, i);
        wB[i] = w_single_bessel(B, k, ell, i);
    }
    if (bisection_set && !system_of_equations_set)
    {
        for (uint i = 0; i < d; i++)
        {
            if (is_diagonal)
            {
                w_precomp[i_integrand][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand][index_bisection[tid] + 1][i] = wB[i];
            }
            else
            {
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid] + 1][i] = wB[i];
            }
        }
    }
    gsl_vector *ce = (col == n_col) ? ce_set[tid] : ce_set_half[tid];
    for (uint i = 0; i < d; i++)
    {
        double p_at_B = 0.0, p_at_A = 0.0;
        for (uint m = 0; m < col; m++)
        {
            double cm = gsl_vector_get(ce, i * col + m);
            p_at_B += cm;
            p_at_A += (m % 2 == 0) ? cm : -cm;
        }
        result += p_at_B * wB[i] - p_at_A * wA[i];
    }
    return result;
}

double pylevin::integrate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    if (A == B)
    {
        return 0.;
    }
    uint tid = omp_get_thread_num();
    double result = 0.0;
    solve_LSE_double(A, B, col, i_integrand, k_1, k_2, ell_1, ell_2);
    double *wA = wA_set[tid].data();
    double *wB = wB_set[tid].data();
    fill_w_double(A, k_1, k_2, ell_1, ell_2, wA);
    fill_w_double(B, k_1, k_2, ell_1, ell_2, wB);
    if (bisection_set && !system_of_equations_set)
    {
        for (uint i = 0; i < d; i++)
        {
            if (is_diagonal)
            {
                w_precomp[i_integrand][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand][index_bisection[tid] + 1][i] = wB[i];
            }
            else
            {
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid] + 1][i] = wB[i];
            }
        }
    }
    gsl_vector *ce = (col == n_col) ? ce_set[tid] : ce_set_half[tid];
    for (uint i = 0; i < d; i++)
    {
        double p_at_B = 0.0, p_at_A = 0.0;
        for (uint m = 0; m < col; m++)
        {
            double cm = gsl_vector_get(ce, i * col + m);
            p_at_B += cm;
            p_at_A += (m % 2 == 0) ? cm : -cm;
        }
        result += p_at_B * wB[i] - p_at_A * wA[i];
    }
    return result;
}

double pylevin::integrate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    if (A == B)
    {
        return 0.;
    }
    uint tid = omp_get_thread_num();
    double result = 0.0;
    solve_LSE_triple(A, B, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    double *wA = wA_set[tid].data();
    double *wB = wB_set[tid].data();
    fill_w_triple(A, k_1, k_2, k_3, ell_1, ell_2, ell_3, wA);
    fill_w_triple(B, k_1, k_2, k_3, ell_1, ell_2, ell_3, wB);
    if (bisection_set && !system_of_equations_set)
    {
        for (uint i = 0; i < d; i++)
        {
            if (is_diagonal)
            {
                w_precomp[i_integrand][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand][index_bisection[tid] + 1][i] = wB[i];
            }
            else
            {
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]][i] = wA[i];
                w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid] + 1][i] = wB[i];
            }
        }
    }
    gsl_vector *ce = (col == n_col) ? ce_set[tid] : ce_set_half[tid];
    for (uint i = 0; i < d; i++)
    {
        double p_at_B = 0.0, p_at_A = 0.0;
        for (uint m = 0; m < col; m++)
        {
            double cm = gsl_vector_get(ce, i * col + m);
            p_at_B += cm;
            p_at_A += (m % 2 == 0) ? cm : -cm;
        }
        result += p_at_B * wB[i] - p_at_A * wA[i];
    }
    return result;
}

double pylevin::integrate_lse_set(double A, double B, uint i_integrand)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    for (uint j = 0; j < n_col; j++)
    {
        gsl_vector_set(F_stacked_set[tid], j, inhomogeneity(map_y_to_x(x_j_set[tid][j], A, B), i_integrand, tid));
    }
    if (is_diagonal)
    {
        gsl_linalg_LU_solve(LU_G_matrix[i_integrand][index_bisection[tid]], permutation[i_integrand][index_bisection[tid]], F_stacked_set[tid], ce_set[tid]);
    }
    else
    {
        gsl_linalg_LU_solve(LU_G_matrix[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], permutation[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]], F_stacked_set[tid], ce_set[tid]);
    }
    for (uint i = 0; i < d; i++)
    {
        double aux_a = 0;
        double aux_b = 0;
        for (uint m = 0; m < n_col; m++)
        {
            if (m % 2 == 0)
            {
                aux_a += gsl_vector_get(ce_set[tid], i * n_col + m);
            }
            else
            {
                aux_a -= gsl_vector_get(ce_set[tid], i * n_col + m);
            }
            aux_b += gsl_vector_get(ce_set[tid], i * n_col + m);
        }
        if (is_diagonal)
        {
            result += aux_b * w_precomp[i_integrand][index_bisection[tid] + 1][i] - aux_a * w_precomp[i_integrand][index_bisection[tid]][i];
        }
        else
        {
            result += aux_b * w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid] + 1][i] - aux_a * w_precomp[i_integrand * size_variables + index_variable[tid]][index_bisection[tid]][i];
        }
    }
    return (B - A) / 2 * result;
}

double pylevin::iterate_single(double A, double B, uint col, uint i_integrand, double k, uint ell, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = 0.0; // integrate_single(A, B, col / 2, i_integrand, k, ell);
    double I_full = integrate_single(A, B, col, i_integrand, k, ell);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, pow(I_full - I_half, 2));
    double result = I_full;
    while (sub <= smax + 1)
    {
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs) && (sqrt(accumulate(error_estimates.begin(), error_estimates.end(), 0.0)) <= GSL_MAX(tol_rel * abs(result), tol_abs)))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                if (is_diagonal)
                {
                    bisection[i_integrand].push_back(x_sub[j]);
                }
                else
                {
                    bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                }
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = std::distance(error_estimates.begin(), std::max_element(error_estimates.begin(), error_estimates.end())) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection for integrand " << i_integrand << " at k " << k << " and ell " << ell << std::endl;
                    for (uint j = 0; j < x_sub.size(); j++)
                    {
                        if (is_diagonal)
                        {
                            bisection[i_integrand].push_back(x_sub[j]);
                        }
                        else
                        {
                            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                        }
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates[i - 1] = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub[i - 1] + x_sub[i]) / 2.);
        double x_subim1_i = (x_sub[i - 1]);
        double x_subi_i = (x_sub[i]);
        double x_subip1_i = (x_sub[i + 1]);
        double old_approx = approximations[i - 1];
        I_half = integrate_single(x_subim1_i, x_subi_i, col / 2, i_integrand, k, ell);
        I_full = integrate_single(x_subim1_i, x_subi_i, col, i_integrand, k, ell);
        approximations[i - 1] = I_full;
        error_estimates[i - 1] = pow(I_full - I_half, 2);
        I_half = integrate_single(x_subi_i, x_subip1_i, col / 2, i_integrand, k, ell);
        I_full = integrate_single(x_subi_i, x_subip1_i, col, i_integrand, k, ell);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, pow(I_full - I_half, 2));
        result = result - old_approx + approximations[i - 1] + I_full;
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k " << k << " and ell " << ell << std::endl;
    }
    error_count = true;
    if (verbose)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        if (is_diagonal)
        {
            bisection[i_integrand].push_back(x_sub[j]);
        }
        else
        {
            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
        }
    }
    return result;
}

double pylevin::iterate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = 0.0;
    double I_full = integrate_double(A, B, col, i_integrand, k_1, k_2, ell_1, ell_2);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, pow(I_full - I_half, 2));
    double result = I_full;
    while (sub <= smax + 1)
    {
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs) && (sqrt(accumulate(error_estimates.begin(), error_estimates.end(), 0.0)) <= GSL_MAX(tol_rel * abs(result), tol_abs)))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                if (is_diagonal)
                {
                    bisection[i_integrand].push_back(x_sub[j]);
                }
                else
                {
                    bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                }
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = std::distance(error_estimates.begin(), std::max_element(error_estimates.begin(), error_estimates.end())) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
                    for (uint j = 0; j < x_sub.size(); j++)
                    {
                        if (is_diagonal)
                        {
                            bisection[i_integrand].push_back(x_sub[j]);
                        }
                        else
                        {
                            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                        }
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates[i - 1] = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub[i - 1] + x_sub[i]) / 2.);
        double x_subim1_i = (x_sub[i - 1]);
        double x_subi_i = (x_sub[i]);
        double x_subip1_i = (x_sub[i + 1]);
        double old_approx = approximations[i - 1];
        I_half = integrate_double(x_subim1_i, x_subi_i, col / 2, i_integrand, k_1, k_2, ell_1, ell_2);
        I_full = integrate_double(x_subim1_i, x_subi_i, col, i_integrand, k_1, k_2, ell_1, ell_2);
        approximations[i - 1] = I_full;
        error_estimates[i - 1] = pow(I_full - I_half, 2);
        I_half = integrate_double(x_subi_i, x_subip1_i, col / 2, i_integrand, k_1, k_2, ell_1, ell_2);
        I_full = integrate_double(x_subi_i, x_subip1_i, col, i_integrand, k_1, k_2, ell_1, ell_2);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, pow(I_full - I_half, 2));
        result = result - old_approx + approximations[i - 1] + I_full;
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
    }
    error_count = true;
    if (verbose)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        if (is_diagonal)
        {
            bisection[i_integrand].push_back(x_sub[j]);
        }
        else
        {
            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
        }
    }
    return result;
}

double pylevin::iterate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = 0.0;
    double I_full = integrate_triple(A, B, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, pow(I_full - I_half, 2));
    double result = I_full;
    while (sub <= smax + 1)
    {
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs) && (sqrt(accumulate(error_estimates.begin(), error_estimates.end(), 0.0)) <= GSL_MAX(tol_rel * abs(result), tol_abs)))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                if (is_diagonal)
                {
                    bisection[i_integrand].push_back(x_sub[j]);
                }
                else
                {
                    bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                }
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = std::distance(error_estimates.begin(), std::max_element(error_estimates.begin(), error_estimates.end())) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
                    for (uint j = 0; j < x_sub.size(); j++)
                    {
                        if (is_diagonal)
                        {
                            bisection[i_integrand].push_back(x_sub[j]);
                        }
                        else
                        {
                            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
                        }
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates[i - 1] = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub[i - 1] + x_sub[i]) / 2.);
        double x_subim1_i = (x_sub[i - 1]);
        double x_subi_i = (x_sub[i]);
        double x_subip1_i = (x_sub[i + 1]);
        double old_approx = approximations[i - 1];
        I_half = integrate_triple(x_subim1_i, x_subi_i, col / 2, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        I_full = integrate_triple(x_subim1_i, x_subi_i, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        approximations[i - 1] = I_full;
        error_estimates[i - 1] = pow(I_full - I_half, 2);
        I_half = integrate_triple(x_subi_i, x_subip1_i, col / 2, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        I_full = integrate_triple(x_subi_i, x_subip1_i, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, pow(I_full - I_half, 2));
        result = result - old_approx + approximations[i - 1] + I_full;
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
    }
    error_count = true;
    if (verbose)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        if (is_diagonal)
        {
            bisection[i_integrand].push_back(x_sub[j]);
        }
        else
        {
            bisection[i_integrand * size_variables + index_variable[tid]].push_back(x_sub[j]);
        }
    }
    return result;
}

double pylevin::levin_integrate_single_bessel(double x_min, double x_max, double k, uint ell, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_single(x_min, x_max, n_col, i_integrand, k, ell, n_sub, speak_to_me);
}

double pylevin::levin_integrate_double_bessel(double x_min, double x_max, double k_1, double k_2, uint ell_1, uint ell_2, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_double(x_min, x_max, n_col, i_integrand, k_1, k_2, ell_1, ell_2, n_sub, speak_to_me);
}

double pylevin::levin_integrate_triple_bessel(double x_min, double x_max, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_triple(x_min, x_max, n_col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3, n_sub, speak_to_me);
}

void pylevin::allocate_variables_for_lse()
{
    if (is_diagonal)
    {
        LU_G_matrix.resize(n_integrand, std::vector<gsl_matrix *>());
        permutation.resize(n_integrand, std::vector<gsl_permutation *>());
        w_precomp.resize(n_integrand, std::vector<std::vector<double>>());
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            for (uint i_bisec = 0; i_bisec < bisection[i_integrand].size() - 1; i_bisec++)
            {
                LU_G_matrix[i_integrand].push_back(gsl_matrix_alloc(d * n_col, d * n_col));
                permutation[i_integrand].push_back(gsl_permutation_alloc(d * n_col));
                w_precomp[i_integrand].push_back(std::vector<double>(d, 0.0));
            }
            w_precomp[i_integrand].push_back(std::vector<double>(d, 0.0));
        }
    }
    else
    {
        LU_G_matrix.resize(n_integrand * size_variables, std::vector<gsl_matrix *>());
        permutation.resize(n_integrand * size_variables, std::vector<gsl_permutation *>());
        w_precomp.resize(n_integrand * size_variables, std::vector<std::vector<double>>());
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            for (uint i_variable = 0; i_variable < size_variables; i_variable++)
            {
                for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                {
                    LU_G_matrix[i_integrand * size_variables + i_variable].push_back(gsl_matrix_alloc(d * n_col, d * n_col));
                    permutation[i_integrand * size_variables + i_variable].push_back(gsl_permutation_alloc(d * n_col));
                    w_precomp[i_integrand * size_variables + i_variable].push_back(std::vector<double>(d, 0.0));
                }
                w_precomp[i_integrand * size_variables + i_variable].push_back(std::vector<double>(d, 0.0));
            }
        }
    }
}

void pylevin::levin_integrate_bessel_single(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k, std::vector<uint> ell, pybind11::array_t<double> &result)
{
    size_variables = x_max.size();
    if (d > 2)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 0 or 1");
    }
    if (is_diagonal)
    {
        if (x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    if (x_min.size() != x_max.size() || x_min.size() != k.size() || x_min.size() != ell.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral[tid] = i_integrand;
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = 0.;
                    for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                    }
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = 0.;
                        index_integral[tid] = i_integrand;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    result.mutable_at(i_variable) = 0.;
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_variable;
                    index_variable[tid] = i_variable;
                    for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable) += integrate_lse_set(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], i_variable);
                    }
                }
            }
        }
    }
    else
    {
        if (!bisection_set)
        {
            if (is_diagonal)
            {
                bisection.resize(n_integrand, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable, i_integrand) = levin_integrate_single_bessel(x_min[i_variable], x_max[i_variable], k[i_variable], ell[i_variable], i_integrand);
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = levin_integrate_single_bessel(x_min[i_variable], x_max[i_variable], k[i_variable], ell[i_variable], i_variable);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            result.mutable_at(i_variable, i_integrand) = levin_integrate_single_bessel(x_min[i_variable], x_max[i_variable], k[i_variable], ell[i_variable], i_integrand);
                        }
                    }
                }
            }
            bisection_set = true;
        }
        else
        {
            allocate_variables_for_lse();
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_integrand;
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        result.mutable_at(i_variable, i_integrand) = 0.;
                        index_variable[tid] = i_variable;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_single(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k[i_variable], ell[i_variable]);
                        }
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        index_integral[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_single(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, i_variable, k[i_variable], ell[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            index_integral[tid] = i_integrand;
                            result.mutable_at(i_variable, i_integrand) = 0.;
                            for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                            {
                                index_bisection[tid] = i_bisec;
                                result.mutable_at(i_variable, i_integrand) += integrate_single(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k[i_variable], ell[i_variable]);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
        }
    }
}

void pylevin::levin_integrate_bessel_double(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<uint> ell_1, std::vector<uint> ell_2, pybind11::array_t<double> &result)
{
    size_variables = x_max.size();
    if (d == 8 || d == 2)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 2 or 3");
    }
    if (is_diagonal)
    {
        if (x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    if (x_min.size() != x_max.size() || x_min.size() != k_1.size() || x_min.size() != k_2.size() || x_min.size() != ell_1.size() || x_min.size() != ell_2.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral[tid] = i_integrand;
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = 0.;
                    for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                    }
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = 0.;
                        index_integral[tid] = i_integrand;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    result.mutable_at(i_variable) = 0.;
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_variable;
                    index_variable[tid] = i_variable;
                    for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable) += integrate_lse_set(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], i_variable);
                    }
                }
            }
        }
    }
    else
    {
        if (!bisection_set)
        {
            if (is_diagonal)
            {
                bisection.resize(n_integrand, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable, i_integrand) = levin_integrate_double_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable], i_integrand);
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = levin_integrate_double_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable], i_variable);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            result.mutable_at(i_variable, i_integrand) = levin_integrate_double_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable], i_integrand);
                        }
                    }
                }
            }
            bisection_set = true;
        }
        else
        {
            allocate_variables_for_lse();
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_integrand;
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        result.mutable_at(i_variable, i_integrand) = 0.;
                        index_variable[tid] = i_variable;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_double(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable]);
                        }
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        index_integral[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_double(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, i_variable, k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            index_integral[tid] = i_integrand;
                            result.mutable_at(i_variable, i_integrand) = 0.;
                            for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                            {
                                index_bisection[tid] = i_bisec;
                                result.mutable_at(i_variable, i_integrand) += integrate_double(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable]);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
        }
    }
}

void pylevin::levin_integrate_bessel_triple(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<double> k_3, std::vector<uint> ell_1, std::vector<uint> ell_2, std::vector<uint> ell_3, pybind11::array_t<double> &result)
{
    size_variables = x_max.size();
    if (d != 8)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 4 or 5");
    }
    if (is_diagonal)
    {
        if (x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    if (x_min.size() != x_max.size() || x_min.size() != k_1.size() || x_min.size() != k_2.size() || x_min.size() != ell_1.size() || x_min.size() != ell_2.size() || x_min.size() != ell_3.size() || x_min.size() != k_3.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral[tid] = i_integrand;
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = 0.;
                    for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                    }
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = 0.;
                        index_integral[tid] = i_integrand;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_lse_set(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], i_integrand);
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    result.mutable_at(i_variable) = 0.;
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_variable;
                    index_variable[tid] = i_variable;
                    for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                    {
                        index_bisection[tid] = i_bisec;
                        result.mutable_at(i_variable) += integrate_lse_set(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], i_variable);
                    }
                }
            }
        }
    }
    else
    {
        if (!bisection_set)
        {
            if (is_diagonal)
            {
                bisection.resize(n_integrand, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable, i_integrand) = levin_integrate_triple_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable], i_integrand);
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = levin_integrate_triple_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable], i_variable);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            result.mutable_at(i_variable, i_integrand) = levin_integrate_triple_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable], i_integrand);
                        }
                    }
                }
            }
            bisection_set = true;
        }
        else
        {
            allocate_variables_for_lse();
            if (size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral[tid] = i_integrand;
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        index_variable[tid] = i_variable;
                        for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable, i_integrand) += integrate_triple(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable]);
                        }
                    }
                }
            }
            else
            {
                if (is_diagonal)
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        index_integral[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_triple(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, i_variable, k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
                            index_integral[tid] = i_integrand;
                            result.mutable_at(i_variable, i_integrand) = 0.;
                            for (uint i_bisec = 0; i_bisec < bisection[i_integrand * size_variables + i_variable].size() - 1; i_bisec++)
                            {
                                index_bisection[tid] = i_bisec;
                                result.mutable_at(i_variable, i_integrand) += integrate_triple(bisection[i_integrand * size_variables + i_variable][i_bisec], bisection[i_integrand * size_variables + i_variable][i_bisec + 1], n_col, i_integrand, k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable]);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
        }
    }
}
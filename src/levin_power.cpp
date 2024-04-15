#include "levin_power.h"
#include <boost/math/special_functions/bessel.hpp>

levin_power::levin_power(uint type_in, std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy, uint nthread)
{
    if (integrand.size() != x.size())
    {
        throw std::range_error("kernel dimension does not match size of chi_cl");
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
    for (uint i = 0; i < N_thread_max; i++)
    {
        index_variable.push_back(0);
        index_integral.push_back(0);
        index_bisection.push_back(0);
    }
    old_handler = gsl_set_error_handler_off();
}

levin_power::~levin_power()
{
    for (uint i = 0; i < n_integrand; i++)
    {
        gsl_spline_free(spline_integrand.at(i));
        gsl_interp_accel_free(acc_integrand.at(i));
    }
    if (system_of_equations_set)
    {
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            for (uint i_variable = 0; i_variable < size_variables; i_variable++)
            {
                for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                {
                    gsl_matrix_free(LU_G_matrix.at(i_integrand).at(i_variable).at(i_bisec));
                    gsl_permutation_free(permutation.at(i_integrand).at(i_variable).at(i_bisec));
                }
            }
        }
    }
    gsl_set_error_handler(old_handler);
}

void levin_power::set_levin(uint n_col_in, uint maximum_number_bisections_in, double relative_accuracy_in, bool super_accurate_in, bool verbose)
{
    n_col = n_col_in;
    maximum_number_subintervals = maximum_number_bisections_in;
    tol_rel = relative_accuracy_in;
    speak_to_me = verbose;
    super_accurate = super_accurate_in;
}

std::vector<std::vector<std::vector<double>>> levin_power::get_bisection()
{
    return bisection;
}

void levin_power::init_splines(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy)
{
    n_integrand = integrand.at(0).size();
    for (uint i_integrand = 0; i_integrand < integrand.at(0).size(); i_integrand++)
    {
        if (!system_of_equations_set && !bisection_set)
        {
            spline_integrand.push_back(gsl_spline_alloc(gsl_interp_cspline, x.size()));
            acc_integrand.push_back(gsl_interp_accel_alloc());
            is_y_log.push_back(false);
            if (!bisection_set)
            {
                bisection.push_back(std::vector<std::vector<double>>());
            }
        }
    }
    for (uint i_integrand = 0; i_integrand < integrand.at(0).size(); i_integrand++)
    {
        if (spline_integrand.size() != integrand.at(0).size())
        {
            std::cout << spline_integrand.size() << " " << integrand.at(0).size() << std::endl;
            throw std::range_error("If you update the integrand, they have to have the same shapes as in the constructor");
        }
        std::vector<double> init_weight(x.size(), 0.0);
        std::vector<double> log_init_weight(x.size(), 0.0);
        if (logy)
        {
            is_y_log.at(i_integrand) = true;
            for (uint i = 0; i < x.size(); i++)
            {
                if (integrand.at(i).at(i_integrand) < 0)
                {
                    is_y_log.at(i_integrand) = false;
                }
            }
        }
        else
        {
            is_y_log.at(i_integrand) = false;
        }
        for (uint i = 0; i < x.size(); i++)
        {
            if (i_integrand == 0)
            {
                if (logx)
                {
                    is_x_log = true;
                    x.at(i) = log(x.at(i));
                }
            }
            if (is_y_log.at(i_integrand))
            {
                init_weight.at(i) = log(integrand.at(i).at(i_integrand));
            }
            else
            {
                init_weight.at(i) = integrand.at(i).at(i_integrand);
            }
        }
        gsl_spline_init(spline_integrand.at(i_integrand), &x[0], &init_weight[0], x.size());
    }
}

void levin_power::update_integrand(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx, bool logy)
{
    init_splines(x, integrand, logx, logy);
}

std::vector<std::vector<double>> levin_power::get_integrand(std::vector<double> x)
{
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(n_integrand));
    for (uint i_x = 0; i_x < x.size(); i_x++)
    {
        double x_value = x.at(i_x);
        if (is_x_log)
        {
            x_value = log(x_value);
        }
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            result.at(i_x).at(i_integrand) = gsl_spline_eval(spline_integrand.at(i_integrand), x_value, acc_integrand.at(i_integrand));
            if (is_y_log.at(i_integrand))
            {
                result.at(i_x).at(i_integrand) = exp(result.at(i_x).at(i_integrand));
            }
        }
    }
    return result;
}

double levin_power::w_single_bessel(double x, double k, uint ell, uint i)
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

double levin_power::w_double_bessel(double x, double k_1, double k_2, uint ell_1, uint ell_2, uint i)
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

double levin_power::w_triple_bessel(double x, double k_1, double k_2, double k_3,  uint ell_1, uint ell_2, uint ell_3, uint i)
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

double levin_power::A_matrix_single(uint i, uint j, double x, double k, uint ell)
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

double levin_power::A_matrix_double(uint i, uint j, double x, double k_1, double k_2, uint ell_1, uint ell_2)
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


double levin_power::A_matrix_triple(uint i, uint j, double x, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
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
        if(i==1 && j==4)
        {
            return -k_2;
        }
        if(j==1 && i==4)
        {
            return k_2;
        }
        if(i==1 && j==6)
        {
            return -k_3;
        }
        if(j==1 && i==6)
        {
            return k_3;
        }
        if(i==2 && j==4)
        {
            return -k_1;
        }
        if(j==2 && i==4)
        {
            return k_1;
        }
        if(i==2 && j==5)
        {
            return -k_3;
        }
        if(j==2 && i==5)
        {
            return k_3;
        }
        if(i==3 && j==5)
        {
            return -k_2;
        }
        if(j==3 && i==5)
        {
            return k_2;
        }
        if(i==3 && j==6)
        {
            return -k_1;
        }
        if(j==3 && i==6)
        {
            return k_1;
        }
        if(i==4 && j==7)
        {
            return -k_3;
        }
        if(j==4 && i==7)
        {
            return k_3;
        }
        if(i==5 && j==7)
        {
            return -k_1;
        }
        if(j==5 && i==7)
        {
            return k_1;
        }
        if(i==6 && j==7)
        {
            return -k_2;
        }
        if(j==6 && i==7)
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
            return (- static_cast<double>(ell_2 + ell_3 + ell_1) - 6.0) / x;
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
        if(i==1 && j==4)
        {
            return -k_2;
        }
        if(j==1 && i==4)
        {
            return k_2;
        }
        if(i==1 && j==6)
        {
            return -k_3;
        }
        if(j==1 && i==6)
        {
            return k_3;
        }
        if(i==2 && j==4)
        {
            return -k_1;
        }
        if(j==2 && i==4)
        {
            return k_1;
        }
        if(i==2 && j==5)
        {
            return -k_3;
        }
        if(j==2 && i==5)
        {
            return k_3;
        }
        if(i==3 && j==5)
        {
            return -k_2;
        }
        if(j==3 && i==5)
        {
            return k_2;
        }
        if(i==3 && j==6)
        {
            return -k_1;
        }
        if(j==3 && i==6)
        {
            return k_1;
        }
        if(i==4 && j==7)
        {
            return -k_3;
        }
        if(j==4 && i==7)
        {
            return k_3;
        }
        if(i==5 && j==7)
        {
            return -k_1;
        }
        if(j==5 && i==7)
        {
            return k_1;
        }
        if(i==6 && j==7)
        {
            return -k_2;
        }
        if(j==6 && i==7)
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
            return (- static_cast<double>(ell_2 + ell_3 + ell_1) - 3.0) / x;
        }
        return 0.0;
    }
    return 0.0;
}

std::vector<double> levin_power::setNodes(double A, double B, uint col)
{
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    for (uint j = 0; j < n; j++)
    {
        x_j[j] = A + j * (B - A) / (n - 1);
    }
    return x_j;
}

double levin_power::basis_function(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 1.0;
    }
    return pow((x - (A + B) / 2) / (B - A), m);
}

double levin_power::basis_function_prime(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 0.0;
    }
    if (m == 1)
    {
        return 1.0 / (B - A);
    }
    return m / (B - A) * pow((x - (A + B) / 2.) / (B - A), (m - 1));
}

double levin_power::inhomogeneity(double x, uint i_integrand)
{
    if (is_x_log)
    {
        x = log(x);
    }
    double result = gsl_spline_eval(spline_integrand.at(i_integrand), x, acc_integrand.at(i_integrand));
    if (is_y_log.at(i_integrand))
    {
        result = exp(result);
    }
    return result;
}

std::vector<double> levin_power::solve_LSE_single(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k, uint ell)
{
    uint tid = omp_get_thread_num();
    double min_sv = 1e-10;
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_single(q, i, x_j[j], k, ell) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s, lu;
    if (bisection_set && !system_of_equations_set)
    {
        gsl_linalg_LU_decomp(matrix_G, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), &s);
        gsl_matrix_memcpy(LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), matrix_G);
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    else
    {
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_matrix_free(matrix_G);
    return result;
}

std::vector<double> levin_power::solve_LSE_double(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    uint tid = omp_get_thread_num();
    double min_sv = 1e-10;
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_double(q, i, x_j[j], k_1, k_2, ell_1, ell_2) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s, lu;
    if (bisection_set && !system_of_equations_set)
    {
        gsl_linalg_LU_decomp(matrix_G, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), &s);
        gsl_matrix_memcpy(LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), matrix_G);
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    else
    {
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_matrix_free(matrix_G);
    return result;
}

std::vector<double> levin_power::solve_LSE_triple(double A, double B, uint col, std::vector<double> x_j, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    uint tid = omp_get_thread_num();
    double min_sv = 1e-10;
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_triple(q, i, x_j[j], k_1, k_2, k_3, ell_1, ell_2, ell_3) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s, lu;
    if (bisection_set && !system_of_equations_set)
    {
        gsl_linalg_LU_decomp(matrix_G, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), &s);
        gsl_matrix_memcpy(LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)), matrix_G);
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    else
    {
        gsl_permutation *P = gsl_permutation_alloc(d * n);
        gsl_linalg_LU_decomp(matrix_G, P, &s);
        lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
        gsl_permutation_free(P);
    }
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_matrix_free(matrix_G);
    return result;
}

double levin_power::p(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint n = (col + 1) / 2;
    n *= 2;
    double result = 0.0;
    for (uint m = 0; m < n; m++)
    {
        result += c[i * n + m] * basis_function(A, B, x, m);
    }
    return result;
}

std::vector<double> levin_power::p_precompute(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint tid = omp_get_thread_num();
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> result(2, 0.0);
    for (uint m = 0; m < n; m++)
    {
        result.at(0) += c[i * n + m] * basis_precomp.at(index_integral.at(tid)).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(m);
        result.at(1) += c[i * n + m] * basis_precomp.at(index_integral.at(tid)).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(m + n);
    }
    return result;
}

double levin_power::integrate_single(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n * d);
    x_j = setNodes(A, B, col);
    c = solve_LSE_single(A, B, col, x_j, i_integrand, k, ell);
    for (uint i = 0; i < d; i++)
    {
        if (bisection_set && !system_of_equations_set)
        {
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i) = w_single_bessel(A, k, ell, i);
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) = w_single_bessel(B, k, ell, i);
        }
        result += p(A, B, i, B, col, c) * w_single_bessel(B, k, ell, i) - p(A, B, i, A, col, c) * w_single_bessel(A, k, ell, i);
    }
    return result;
}

double levin_power::integrate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n * d);
    x_j = setNodes(A, B, col);
    c = solve_LSE_double(A, B, col, x_j, i_integrand, k_1, k_2, ell_1, ell_2);
    for (uint i = 0; i < d; i++)
    {
        if (bisection_set && !system_of_equations_set)
        {
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i) = w_double_bessel(A, k_1, k_2, ell_1, ell_2, i);
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) = w_double_bessel(B, k_1, k_2, ell_1, ell_2, i);
        }
        result += p(A, B, i, B, col, c) * w_double_bessel(B, k_1, k_2, ell_1, ell_2, i) - p(A, B, i, A, col, c) * w_double_bessel(A, k_1, k_2, ell_1, ell_2, i);
    }
    return result;
}

double levin_power::integrate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n * d);
    x_j = setNodes(A, B, col);
    c = solve_LSE_triple(A, B, col, x_j, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    for (uint i = 0; i < d; i++)
    {
        if (bisection_set && !system_of_equations_set)
        {
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i) = w_triple_bessel(A, k_1, k_2, k_3, ell_1, ell_2, ell_3, i);
            w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) = w_triple_bessel(B, k_1, k_2, k_3,  ell_1, ell_2, ell_3, i);
        }
        result += p(A, B, i, B, col, c) * w_triple_bessel(B, k_1, k_2, k_3, ell_1, ell_2, ell_3, i) - p(A, B, i, A, col, c) * w_triple_bessel(A, k_1, k_2, k_3, ell_1, ell_2, ell_3, i);
    }
    return result;
}

double levin_power::integrate_lse_set_single(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    x_j = setNodes(A, B, col);
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *ce = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(matrix_G, LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, ce);
    gsl_permutation_free(P);
    gsl_matrix_free(matrix_G);
    std::vector<double> c(n * d);
    for (uint j = 0; j < d * n; j++)
    {
        c.at(j) = gsl_vector_get(ce, j);
    }
    std::vector<double> aux(2, 0);
    for (uint i = 0; i < d; i++)
    {
        aux = p_precompute(A, B, i, B, col, c);
        result += aux.at(1) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) - aux.at(0) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i);
    }
    gsl_vector_free(F_stacked);
    gsl_vector_free(ce);
    return result;
}

double levin_power::integrate_lse_set_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    x_j = setNodes(A, B, col);
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *ce = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(matrix_G, LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, ce);
    gsl_permutation_free(P);
    gsl_matrix_free(matrix_G);
    std::vector<double> c(n * d);
    for (uint j = 0; j < d * n; j++)
    {
        c.at(j) = gsl_vector_get(ce, j);
    }
    std::vector<double> aux(2, 0);
    for (uint i = 0; i < d; i++)
    {
        aux = p_precompute(A, B, i, B, col, c);
        result += aux.at(1) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) - aux.at(0) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i);
    }
    gsl_vector_free(F_stacked);
    gsl_vector_free(ce);
    return result;
}

double levin_power::integrate_lse_set_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    uint tid = omp_get_thread_num();
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    x_j = setNodes(A, B, col);
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *ce = gsl_vector_alloc(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, inhomogeneity(x_j[j], i_integrand));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(matrix_G, LU_G_matrix.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_permutation_memcpy(P, permutation.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)));
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, ce);
    gsl_permutation_free(P);
    gsl_matrix_free(matrix_G);
    std::vector<double> c(n * d);
    for (uint j = 0; j < d * n; j++)
    {
        c.at(j) = gsl_vector_get(ce, j);
    }
    std::vector<double> aux(2, 0);
    for (uint i = 0; i < d; i++)
    {
        aux = p_precompute(A, B, i, B, col, c);
        result += aux.at(1) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid) + 1).at(i) - aux.at(0) * w_precomp.at(i_integrand).at(index_variable.at(tid)).at(index_bisection.at(tid)).at(i);
    }
    gsl_vector_free(F_stacked);
    gsl_vector_free(ce);
    return result;
}


double levin_power::iterate_single(double A, double B, uint col, uint i_integrand, double k, uint ell, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_single(A, B, col / 2, i_integrand, k, ell);
    double I_full = integrate_single(A, B, col, i_integrand, k, ell);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
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
                        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        double x_subim1_i = (x_sub.at(i - 1));
        double x_subi_i = (x_sub.at(i));
        double x_subip1_i = (x_sub.at(i + 1));
        I_half = integrate_single(x_subim1_i, x_subi_i, col / 2, i_integrand, k, ell);
        I_full = integrate_single(x_subim1_i, x_subi_i, col, i_integrand, k, ell);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_single(x_subi_i, x_subip1_i, col / 2, i_integrand, k, ell);
        I_full = integrate_single(x_subi_i, x_subip1_i, col, i_integrand, k, ell);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k " << k << " and ell " << ell << std::endl;
    }
    error_count = true;
    if (error_count == true)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
    }
    return result;
}

double levin_power::iterate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_double(A, B, col / 2, i_integrand, k_1, k_2, ell_1, ell_2);
    double I_full = integrate_double(A, B, col, i_integrand, k_1, k_2, ell_1, ell_2);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
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
                        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        double x_subim1_i = (x_sub.at(i - 1));
        double x_subi_i = (x_sub.at(i));
        double x_subip1_i = (x_sub.at(i + 1));
        I_half = integrate_double(x_subim1_i, x_subi_i, col / 2, i_integrand, k_1, k_2, ell_1, ell_2);
        I_full = integrate_double(x_subim1_i, x_subi_i, col, i_integrand, k_1, k_2, ell_1, ell_2);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_double(x_subi_i, x_subip1_i, col / 2, i_integrand, k_1, k_2, ell_1, ell_2);
        I_full = integrate_double(x_subi_i, x_subip1_i, col, i_integrand, k_1, k_2, ell_1, ell_2);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
    }
    error_count = true;
    if (error_count == true)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
    }
    return result;
}

double levin_power::iterate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint smax, bool verbose)
{
    uint tid = omp_get_thread_num();
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_triple(A, B, col / 2, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    double I_full = integrate_triple(A, B, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs))
        {
            for (uint j = 0; j < x_sub.size(); j++)
            {
                bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
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
                        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
                    }
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        double x_subim1_i = (x_sub.at(i - 1));
        double x_subi_i = (x_sub.at(i));
        double x_subip1_i = (x_sub.at(i + 1));
        I_half = integrate_triple(x_subim1_i, x_subi_i, col / 2, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        I_full = integrate_triple(x_subim1_i, x_subi_i, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_triple(x_subi_i, x_subip1_i, col / 2, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        I_full = integrate_triple(x_subi_i, x_subip1_i, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2 << std::endl;
    }
    error_count = true;
    if (error_count == true)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    for (uint j = 0; j < x_sub.size(); j++)
    {
        bisection.at(i_integrand).at(index_variable.at(tid)).push_back(x_sub.at(j));
    }
    return result;
}

double levin_power::levin_integrate_single_bessel(double x_min, double x_max, double k, uint ell, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_single(x_min, x_max, n_col, i_integrand, k, ell, n_sub, speak_to_me);
}

double levin_power::levin_integrate_double_bessel(double x_min, double x_max, double k_1, double k_2, uint ell_1, uint ell_2, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_double(x_min, x_max, n_col, i_integrand, k_1, k_2, ell_1, ell_2, n_sub, speak_to_me);
}

double levin_power::levin_integrate_triple_bessel(double x_min, double x_max, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint i_integrand)
{
    uint n_sub = maximum_number_subintervals;
    return iterate_triple(x_min, x_max, n_col, i_integrand, k_1, k_2,k_3, ell_1, ell_2, ell_3, n_sub, speak_to_me);
}

std::vector<std::vector<double>> levin_power::levin_integrate_bessel_single(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k, std::vector<uint> ell, bool diagonal)
{
    if(diagonal)
    {
        if(x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    uint n = (n_col + 1) / 2;
    n *= 2;
    std::vector<std::vector<double>> result(x_min.size(), std::vector<double>(n_integrand, 0.0));
    if (x_min.size() != x_max.size() || x_min.size() != k.size() || x_min.size() != ell.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (x_min.size() < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral.at(tid) = i_integrand;
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_single(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k.at(i_variable), ell.at(i_variable));
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    index_integral.at(tid) = i_integrand;
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_single(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k.at(i_variable), ell.at(i_variable));
                    }
                }
            }
        }
        return result;
    }
    else
    {
        if (!bisection_set)
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    if(diagonal && i_variable != i_integrand)
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>(2,0.0));
                    }
                    else
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>());
                    }
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }    
                        result.at(i_variable).at(i_integrand) = levin_integrate_single_bessel(x_min.at(i_variable), x_max.at(i_variable), k.at(i_variable), ell.at(i_variable), i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        result.at(i_variable).at(i_integrand) = levin_integrate_single_bessel(x_min.at(i_variable), x_max.at(i_variable), k.at(i_variable), ell.at(i_variable), i_integrand);
                    }
                }
            }
            bisection_set = true;
            return result;
        }
        else
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                LU_G_matrix.push_back(std::vector<std::vector<gsl_matrix *>>());
                permutation.push_back(std::vector<std::vector<gsl_permutation *>>());
                w_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                basis_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    LU_G_matrix.at(i_integrand).push_back(std::vector<gsl_matrix *>());
                    permutation.at(i_integrand).push_back(std::vector<gsl_permutation *>());
                    w_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    basis_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        
                        LU_G_matrix.at(i_integrand).at(i_variable).push_back(gsl_matrix_alloc(d * n, d * n));
                        permutation.at(i_integrand).at(i_variable).push_back(gsl_permutation_alloc(d * n));
                        w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2, 0.0));
                        basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                    }
                    w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2, 0.0));
                    basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral.at(tid) = i_integrand;
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_single(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k.at(i_variable), ell.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        index_integral.at(tid) = i_integrand;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_single(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k.at(i_variable), ell.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
            size_variables = x_max.size();
            return result;
        }
    }
}

std::vector<std::vector<double>> levin_power::levin_integrate_bessel_double(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<uint> ell_1, std::vector<uint> ell_2, bool diagonal)
{
    if(diagonal)
    {
        if(x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    uint n = (n_col + 1) / 2;
    n *= 2;
    std::vector<std::vector<double>> result(x_min.size(), std::vector<double>(n_integrand, 0.0));
    if (x_min.size() != x_max.size() || x_min.size() != k_1.size() || x_min.size() != k_2.size() || x_min.size() != ell_1.size() || x_min.size() != ell_2.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (x_min.size() < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral.at(tid) = i_integrand;
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }   
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_double(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable));
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    index_integral.at(tid) = i_integrand;
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_double(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable));
                    }
                }
            }
        }
        return result;
    }
    else
    {
        if (!bisection_set)
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    if(diagonal && i_variable != i_integrand)
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>(2,0.0));
                    }
                    else
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>());
                    }  
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        result.at(i_variable).at(i_integrand) = levin_integrate_double_bessel(x_min.at(i_variable), x_max.at(i_variable), k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        result.at(i_variable).at(i_integrand) = levin_integrate_double_bessel(x_min.at(i_variable), x_max.at(i_variable), k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), i_integrand);
                    }
                }
            }
            bisection_set = true;
            return result;
        }
        else
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                LU_G_matrix.push_back(std::vector<std::vector<gsl_matrix *>>());
                permutation.push_back(std::vector<std::vector<gsl_permutation *>>());
                w_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                basis_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    LU_G_matrix.at(i_integrand).push_back(std::vector<gsl_matrix *>());
                    permutation.at(i_integrand).push_back(std::vector<gsl_permutation *>());
                    w_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    basis_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        LU_G_matrix.at(i_integrand).at(i_variable).push_back(gsl_matrix_alloc(d * n, d * n));
                        permutation.at(i_integrand).at(i_variable).push_back(gsl_permutation_alloc(d * n));
                        w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(4, 0.0));
                        basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                    }
                    w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(4, 0.0));
                    basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral.at(tid) = i_integrand;
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_double(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        index_integral.at(tid) = i_integrand;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_double(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
            size_variables = x_max.size();
            return result;
        }
    }
}


std::vector<std::vector<double>> levin_power::levin_integrate_bessel_triple(std::vector<double> x_min, std::vector<double> x_max, std::vector<double> k_1, std::vector<double> k_2, std::vector<double> k_3, std::vector<uint> ell_1, std::vector<uint> ell_2, std::vector<uint> ell_3, bool diagonal)
{
    if(diagonal)
    {
        if(x_min.size() != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
    }
    uint n = (n_col + 1) / 2;
    n *= 2;
    std::vector<std::vector<double>> result(x_min.size(), std::vector<double>(n_integrand, 0.0));
    if (x_min.size() != x_max.size() || x_min.size() != k_1.size() || x_min.size() != k_2.size() || x_min.size() != ell_1.size() || x_min.size() != ell_2.size() || x_min.size() != ell_3.size() || x_min.size() != k_3.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    if (system_of_equations_set)
    {
        if (x_min.size() < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                index_integral.at(tid) = i_integrand;
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_triple(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable));
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    index_integral.at(tid) = i_integrand;
                    index_variable.at(tid) = i_variable;
                    if(diagonal && i_variable != i_integrand)
                    {
                        continue;
                    }
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        index_bisection.at(tid) = i_bisec;
                        result.at(i_variable).at(i_integrand) += integrate_lse_set_triple(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable));
                    }
                }
            }
        }
        return result;
    }
    else
    {
        if (!bisection_set)
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    if(diagonal && i_variable != i_integrand)
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>(2,0.0));
                    }
                    else
                    {
                        bisection.at(i_integrand).push_back(std::vector<double>());
                    }
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        result.at(i_variable).at(i_integrand) = levin_integrate_triple_bessel(x_min.at(i_variable), x_max.at(i_variable), k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable), i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        result.at(i_variable).at(i_integrand) = levin_integrate_triple_bessel(x_min.at(i_variable), x_max.at(i_variable), k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable), i_integrand);
                    }
                }
            }
            bisection_set = true;
            return result;
        }
        else
        {
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                LU_G_matrix.push_back(std::vector<std::vector<gsl_matrix *>>());
                permutation.push_back(std::vector<std::vector<gsl_permutation *>>());
                w_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                basis_precomp.push_back(std::vector<std::vector<std::vector<double>>>());
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    LU_G_matrix.at(i_integrand).push_back(std::vector<gsl_matrix *>());
                    permutation.at(i_integrand).push_back(std::vector<gsl_permutation *>());
                    w_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    basis_precomp.at(i_integrand).push_back(std::vector<std::vector<double>>());
                    for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                    {
                        LU_G_matrix.at(i_integrand).at(i_variable).push_back(gsl_matrix_alloc(d * n, d * n));
                        permutation.at(i_integrand).at(i_variable).push_back(gsl_permutation_alloc(d * n));
                        w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(d, 0.0));
                        basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                    }
                    w_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(d, 0.0));
                    basis_precomp.at(i_integrand).at(i_variable).push_back(std::vector<double>(2 * n, 1.0));
                }
            }
            if (x_min.size() < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
                    index_integral.at(tid) = i_integrand;
                    for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                    {
                        index_variable.at(tid) = i_variable;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_triple(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max)
                for (uint i_variable = 0; i_variable < x_max.size(); i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable.at(tid) = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        index_integral.at(tid) = i_integrand;
                        if(diagonal && i_variable != i_integrand)
                        {
                            continue;
                        }
                        for (uint i_bisec = 0; i_bisec < bisection.at(i_integrand).at(i_variable).size() - 1; i_bisec++)
                        {
                            index_bisection.at(tid) = i_bisec;
                            result.at(i_variable).at(i_integrand) += integrate_triple(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), n_col, i_integrand, k_1.at(i_variable), k_2.at(i_variable), k_3.at(i_variable), ell_1.at(i_variable), ell_2.at(i_variable), ell_3.at(i_variable));
                            for (uint i_col = 0; i_col < n; i_col++)
                            {
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec), i_col);
                                basis_precomp.at(i_integrand).at(i_variable).at(i_bisec).at(n + i_col) = basis_function(bisection.at(i_integrand).at(i_variable).at(i_bisec), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), bisection.at(i_integrand).at(i_variable).at(i_bisec + 1), i_col);
                            }
                        }
                    }
                }
            }
            system_of_equations_set = true;
            size_variables = x_max.size();
            return result;
        }
    }
}
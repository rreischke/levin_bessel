#include "pylevin.h"
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>

namespace
{
std::uint64_t double_bit_pattern(double value)
{
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void fill_chebyshev_basis(double x, uint col, double *values, double *derivatives)
{
    if (col == 0)
    {
        return;
    }

    values[0] = 1.0;
    derivatives[0] = 0.0;
    if (col == 1)
    {
        return;
    }

    values[1] = x;
    derivatives[1] = 1.0;
    for (uint m = 2; m < col; ++m)
    {
        values[m] = 2.0 * x * values[m - 1] - values[m - 2];
        derivatives[m] = 2.0 * values[m - 1] + 2.0 * x * derivatives[m - 1] - derivatives[m - 2];
    }
}

std::vector<double> multiply_chebyshev_polynomials(const std::vector<double> &a,
                                                    const std::vector<double> &b)
{
    if (a.empty() || b.empty())
    {
        return {};
    }
    std::vector<double> result(a.size() + b.size() - 1, 0.0);
    for (uint i = 0; i < a.size(); ++i)
    {
        for (uint j = 0; j < b.size(); ++j)
        {
            const double value = 0.5 * a[i] * b[j];
            result[i + j] += value;
            result[(i > j) ? (i - j) : (j - i)] += value;
        }
    }
    return result;
}

uint alias_chebyshev_mode(uint mode, uint number_of_modes)
{
    const uint endpoint_mode = number_of_modes - 1;
    const uint period = 2 * endpoint_mode;
    const uint folded = mode % period;
    return (folded <= endpoint_mode) ? folded : (period - folded);
}

void fft_radix2(std::vector<std::complex<double>> &values, bool inverse)
{
    const size_t count = values.size();
    for (size_t i = 1, j = 0; i < count; ++i)
    {
        size_t bit = count >> 1;
        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(values[i], values[j]);
        }
    }

    for (size_t length = 2; length <= count; length <<= 1)
    {
        const double angle = (inverse ? 2.0 : -2.0) * M_PI / length;
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (size_t first = 0; first < count; first += length)
        {
            std::complex<double> factor(1.0, 0.0);
            const size_t half = length / 2;
            for (size_t j = 0; j < half; ++j)
            {
                const std::complex<double> even = values[first + j];
                const std::complex<double> odd = values[first + j + half] * factor;
                values[first + j] = even + odd;
                values[first + j + half] = even - odd;
                factor *= root;
            }
        }
        if (length == count)
        {
            break;
        }
    }

    if (inverse)
    {
        const double scale = 1.0 / count;
        for (std::complex<double> &value : values)
        {
            value *= scale;
        }
    }
}

void make_dct_plan(uint count, size_t &convolution_size,
                   std::vector<std::complex<double>> &chirp,
                   std::vector<std::complex<double>> &kernel_fft)
{
    if (count < 2)
    {
        convolution_size = 0;
        chirp.clear();
        kernel_fft.clear();
        return;
    }
    const size_t transform_size = 2 * static_cast<size_t>(count - 1);
    size_t required_size = 2 * transform_size - 1;
    convolution_size = 1;
    while (convolution_size < required_size)
    {
        convolution_size <<= 1;
    }
    chirp.resize(transform_size);
    kernel_fft.assign(convolution_size, std::complex<double>(0.0, 0.0));
    for (size_t index = 0; index < transform_size; ++index)
    {
        const double phase = M_PI * static_cast<double>(index) * index / transform_size;
        chirp[index] = std::complex<double>(std::cos(phase), -std::sin(phase));
    }
    for (int offset = -static_cast<int>(transform_size - 1);
         offset <= static_cast<int>(transform_size - 1); ++offset)
    {
        const double phase = M_PI * static_cast<double>(offset) * offset / transform_size;
        const std::complex<double> value(std::cos(phase), std::sin(phase));
        const size_t index = (offset < 0)
                                 ? convolution_size - static_cast<size_t>(-offset)
                                 : static_cast<size_t>(offset);
        kernel_fft[index] = value;
    }
    fft_radix2(kernel_fft, false);
}

void chebyshev_coefficients_from_lobatto_values(
    const std::vector<double> &values, std::vector<double> &coefficients,
    size_t convolution_size, const std::vector<std::complex<double>> &chirp,
    const std::vector<std::complex<double>> &kernel_fft)
{
    const uint count = static_cast<uint>(values.size());
    coefficients.assign(count, 0.0);
    if (count < 2)
    {
        if (count == 1)
        {
            coefficients[0] = values[0];
        }
        return;
    }

    const size_t degree = count - 1;
    if (count <= 32 || convolution_size == 0 || chirp.size() != 2 * degree ||
        kernel_fft.size() != convolution_size)
    {
        for (uint mode = 0; mode < count; ++mode)
        {
            double sum = 0.5 * values[0] +
                         0.5 * values[degree] * ((mode % 2 == 0) ? 1.0 : -1.0);
            for (uint node = 1; node < degree; ++node)
            {
                sum += values[node] * std::cos(M_PI * mode * node / degree);
            }
            coefficients[mode] = (2.0 / degree) * sum *
                                 ((mode == 0 || mode == degree) ? 0.5 : 1.0);
        }
        return;
    }

    const size_t transform_size = 2 * degree;
    std::vector<std::complex<double>> work(convolution_size,
                                           std::complex<double>(0.0, 0.0));
    work[0] = values[0];
    work[degree] = values[degree] * chirp[degree];
    for (size_t node = 1; node < degree; ++node)
    {
        const std::complex<double> value = values[node] * chirp[node];
        work[node] = value;
        work[transform_size - node] = values[node] * chirp[transform_size - node];
    }

    fft_radix2(work, false);
    for (size_t index = 0; index < convolution_size; ++index)
    {
        work[index] *= kernel_fft[index];
    }
    fft_radix2(work, true);
    const double scale = 1.0 / degree;
    for (uint mode = 0; mode < count; ++mode)
    {
        const double endpoint_scale = (mode == 0 || mode == degree) ? 0.5 : 1.0;
        coefficients[mode] = (work[mode] * chirp[mode]).real() * scale * endpoint_scale;
    }
}

size_t band_stride(uint lower_bandwidth, uint upper_bandwidth)
{
    return static_cast<size_t>(2 * lower_bandwidth + upper_bandwidth + 1);
}

size_t band_index(uint row, uint column, uint lower_bandwidth, size_t stride)
{
    const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(column) -
                                  static_cast<std::ptrdiff_t>(row) + lower_bandwidth;
    return static_cast<size_t>(row) * stride + static_cast<size_t>(offset);
}

bool banded_lu_factor(std::vector<double> &matrix, uint size, uint lower_bandwidth,
                      uint upper_bandwidth)
{
    if (size == 0)
    {
        return true;
    }
    const size_t stride = band_stride(lower_bandwidth, upper_bandwidth);
    if (matrix.size() != static_cast<size_t>(size) * stride)
    {
        return false;
    }
    double matrix_scale = 0.0;
    for (double value : matrix)
    {
        matrix_scale = std::max(matrix_scale, std::abs(value));
    }
    if (!std::isfinite(matrix_scale) || matrix_scale == 0.0)
    {
        return false;
    }
    const double pivot_floor = 32.0 * std::numeric_limits<double>::epsilon() * matrix_scale;
    for (uint column = 0; column < size; ++column)
    {
        // No row pivoting is required by the regular high-frequency cases for
        // which this spectral formulation is intended. If a diagonal pivot is
        // unsafe, report failure so the caller can use the pivoted dense GSL
        // solver instead.
        const double pivot_abs = std::abs(matrix[band_index(column, column, lower_bandwidth, stride)]);
        if (!std::isfinite(pivot_abs) || pivot_abs <= pivot_floor)
        {
            return false;
        }
        const double diagonal = matrix[band_index(column, column, lower_bandwidth, stride)];
        const uint last_row = std::min(size - 1, column + lower_bandwidth);
        const uint last_column = std::min(size - 1, column + upper_bandwidth + lower_bandwidth);
        for (uint row = column + 1; row <= last_row; ++row)
        {
            double &entry = matrix[band_index(row, column, lower_bandwidth, stride)];
            if (entry == 0.0)
            {
                continue;
            }
            const double multiplier = entry / diagonal;
            entry = multiplier;
            for (uint j = column + 1; j <= last_column; ++j)
            {
                if (j + lower_bandwidth < row)
                {
                    continue;
                }
                matrix[band_index(row, j, lower_bandwidth, stride)] -=
                    multiplier * matrix[band_index(column, j, lower_bandwidth, stride)];
            }
        }
    }
    return true;
}

bool banded_lu_solve(const std::vector<double> &matrix, uint size, uint lower_bandwidth,
                     uint upper_bandwidth, std::vector<double> &rhs)
{
    if (rhs.size() != size)
    {
        return false;
    }
    const size_t stride = band_stride(lower_bandwidth, upper_bandwidth);
    for (uint column = 0; column < size; ++column)
    {
        const uint last_row = std::min(size - 1, column + lower_bandwidth);
        for (uint row = column + 1; row <= last_row; ++row)
        {
            rhs[row] -= matrix[band_index(row, column, lower_bandwidth, stride)] * rhs[column];
        }
    }
    for (int row = static_cast<int>(size) - 1; row >= 0; --row)
    {
        const uint first_column = static_cast<uint>(row) + 1;
        const uint last_column = std::min(size - 1, static_cast<uint>(row) + upper_bandwidth + lower_bandwidth);
        for (uint column = first_column; column <= last_column; ++column)
        {
            rhs[row] -= matrix[band_index(static_cast<uint>(row), column,
                                          lower_bandwidth, stride)] * rhs[column];
        }
        const double diagonal = matrix[band_index(static_cast<uint>(row), static_cast<uint>(row),
                                                  lower_bandwidth, stride)];
        if (!std::isfinite(diagonal) || diagonal == 0.0)
        {
            return false;
        }
        rhs[row] /= diagonal;
        if (!std::isfinite(rhs[row]))
        {
            return false;
        }
    }
    return true;
}

bool banded_solve_residual_ok(const std::vector<double> &matrix, uint size,
                              uint lower_bandwidth, uint upper_bandwidth,
                              const std::vector<double> &solution,
                              const std::vector<double> &right_hand_side)
{
    double maximum_residual = 0.0;
    double maximum_rhs = 0.0;
    double maximum_row_scale = 0.0;
    double maximum_solution = 0.0;
    const size_t stride = band_stride(lower_bandwidth, upper_bandwidth);
    for (uint row = 0; row < size; ++row)
    {
        double value = 0.0;
        double row_scale = 0.0;
        const uint first_column = (row > lower_bandwidth) ? row - lower_bandwidth : 0;
        const uint last_column = std::min(size - 1, row + upper_bandwidth);
        for (uint column = first_column; column <= last_column; ++column)
        {
            const double coefficient = matrix[band_index(row, column, lower_bandwidth, stride)];
            value += coefficient * solution[column];
            row_scale += std::abs(coefficient);
        }
        maximum_residual = std::max(maximum_residual, std::abs(value - right_hand_side[row]));
        maximum_rhs = std::max(maximum_rhs, std::abs(right_hand_side[row]));
        maximum_row_scale = std::max(maximum_row_scale, row_scale);
        maximum_solution = std::max(maximum_solution, std::abs(solution[row]));
    }
    const double scale = maximum_rhs + maximum_row_scale * maximum_solution;
    return std::isfinite(maximum_residual) &&
           maximum_residual <= 1e-10 * std::max(scale, std::numeric_limits<double>::min());
}

bool dense_pivot_solve(std::vector<double> &matrix, uint size, std::vector<double> &rhs)
{
    if (matrix.size() != size * size || rhs.size() != size)
    {
        return false;
    }
    double matrix_scale = 0.0;
    for (double value : matrix)
    {
        matrix_scale = std::max(matrix_scale, std::abs(value));
    }
    const double pivot_floor = 32.0 * std::numeric_limits<double>::epsilon() * matrix_scale;
    for (uint column = 0; column < size; ++column)
    {
        uint pivot_row = column;
        double pivot_abs = std::abs(matrix[column * size + column]);
        for (uint row = column + 1; row < size; ++row)
        {
            const double candidate = std::abs(matrix[row * size + column]);
            if (candidate > pivot_abs)
            {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if (!std::isfinite(pivot_abs) || pivot_abs <= pivot_floor)
        {
            return false;
        }
        if (pivot_row != column)
        {
            for (uint j = column; j < size; ++j)
            {
                std::swap(matrix[column * size + j], matrix[pivot_row * size + j]);
            }
            std::swap(rhs[column], rhs[pivot_row]);
        }
        for (uint row = column + 1; row < size; ++row)
        {
            const double multiplier = matrix[row * size + column] / matrix[column * size + column];
            matrix[row * size + column] = 0.0;
            for (uint j = column + 1; j < size; ++j)
            {
                matrix[row * size + j] -= multiplier * matrix[column * size + j];
            }
            rhs[row] -= multiplier * rhs[column];
        }
    }
    for (int row = static_cast<int>(size) - 1; row >= 0; --row)
    {
        for (uint column = static_cast<uint>(row) + 1; column < size; ++column)
        {
            rhs[row] -= matrix[static_cast<uint>(row) * size + column] * rhs[column];
        }
        rhs[row] /= matrix[static_cast<uint>(row) * size + static_cast<uint>(row)];
        if (!std::isfinite(rhs[row]))
        {
            return false;
        }
    }
    return true;
}

std::vector<double> interpolatory_weights(uint col)
{
    std::vector<double> matrix(static_cast<size_t>(col) * col);
    std::vector<double> moments(col, 0.0), values(col), derivatives(col);
    const double norm = -1.0 / cos(M_PI * ((1. + 2 * col) / (2 * col)));
    for (uint j = 0; j < col; j++)
    {
        const double y = norm * cos((2. * (j + 1) - 1) / (2. * col) * M_PI + M_PI);
        fill_chebyshev_basis(y, col, values.data(), derivatives.data());
        for (uint m = 0; m < col; m++)
        {
            matrix[m * col + j] = values[m];
        }
    }
    for (uint m = 0; m < col; m += 2)
    {
        moments[m] = 2.0 / (1.0 - static_cast<double>(m) * m);
    }
    if (!dense_pivot_solve(matrix, col, moments))
    {
        throw std::runtime_error("could not compute the interpolatory quadrature weights");
    }
    return moments;
}

std::vector<uint> bessel_orders(const pylevin::array_int &ell)
{
    std::vector<uint> orders(ell.size());
    for (uint i = 0; i < orders.size(); i++)
    {
        if (ell.data()[i] < 0)
        {
            throw std::range_error("Bessel orders must be non-negative");
        }
        orders[i] = ell.data()[i];
    }
    return orders;
}
}

std::vector<uint> pylevin::integrand_indices(const std::optional<array_int> &integrand) const
{
    // Integrand of each variable in diagonal mode: the given indices, or the variable itself.
    std::vector<uint> indices(size_variables);
    if (!integrand)
    {
        if (is_diagonal && size_variables != n_integrand)
        {
            throw std::range_error("The number of integrands must match the number of variables at which the integral is called in diagonal mode");
        }
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    if (!is_diagonal)
    {
        throw std::range_error("integrand indices can only be given in diagonal mode");
    }
    if (integrand->size() != size_variables)
    {
        throw std::range_error("sizes of all arguments must match");
    }
    for (uint i = 0; i < size_variables; i++)
    {
        const std::int64_t index = integrand->data()[i];
        if (index < 0 || index >= n_integrand)
        {
            throw std::range_error("integrand indices must lie in [0, number of integrands)");
        }
        indices[i] = index;
    }
    return indices;
}

pylevin::pylevin(uint type_in, const array_double &x, const array_double &integrand, bool logx, bool logy, uint nthread, bool diagonal)
{
    is_diagonal = diagonal;
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
    index_bisection.resize(N_thread_max);
    set_levin(8, 32, 1e-4, false, false, 0.0);
}

pylevin::~pylevin()
{
    for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
    {
        if (F_stacked_set[i_thread]) gsl_vector_free(F_stacked_set[i_thread]);
        if (F_stacked_set_half[i_thread]) gsl_vector_free(F_stacked_set_half[i_thread]);
        if (ce_set[i_thread]) gsl_vector_free(ce_set[i_thread]);
        if (ce_set_half[i_thread]) gsl_vector_free(ce_set_half[i_thread]);
        if (matrix_G_set[i_thread]) gsl_matrix_free(matrix_G_set[i_thread]);
        if (matrix_G_set_half[i_thread]) gsl_matrix_free(matrix_G_set_half[i_thread]);
        if (P_set[i_thread]) gsl_permutation_free(P_set[i_thread]);
        if (P_set_half[i_thread]) gsl_permutation_free(P_set_half[i_thread]);
    }
}

void pylevin::set_levin(uint n_col_in, uint maximum_number_bisections_in, double relative_accuracy_in, bool super_accurate_in, bool verbose, double tol_abs_in, bool use_banded_levin_in, double max_ratio_in, double non_oscillatory_in)
{
    tol_abs = tol_abs_in;
    max_ratio = max_ratio_in;
    non_oscillatory = non_oscillatory_in;
    n_col = (n_col_in + 1) / 2;
    n_col *= 2;
    maximum_number_subintervals = maximum_number_bisections_in;
    tol_rel = relative_accuracy_in;
    speak_to_me = verbose;
    super_accurate = super_accurate_in;
    // The packed solver pays off at higher coefficient counts. Keep the
    // pivoted dense solver for smaller systems.
    use_banded_levin = use_banded_levin_in && type < 2 && n_col >= 64;
    if (system_of_equations_set)
    {
        weights_precomp.clear();
        spline_moments.clear();
        grid_nodes.clear();
        sample_weights.clear();
        system_of_equations_set = false;
    }
    // Free previously allocated per-thread GSL objects to avoid leaks on re-call
    if (!matrix_G_set.empty())
    {
        for (uint i_thread = 0; i_thread < matrix_G_set.size(); i_thread++)
        {
            if (F_stacked_set[i_thread]) gsl_vector_free(F_stacked_set[i_thread]);
            if (F_stacked_set_half[i_thread]) gsl_vector_free(F_stacked_set_half[i_thread]);
            if (ce_set[i_thread]) gsl_vector_free(ce_set[i_thread]);
            if (ce_set_half[i_thread]) gsl_vector_free(ce_set_half[i_thread]);
            if (matrix_G_set[i_thread]) gsl_matrix_free(matrix_G_set[i_thread]);
            if (matrix_G_set_half[i_thread]) gsl_matrix_free(matrix_G_set_half[i_thread]);
            if (P_set[i_thread]) gsl_permutation_free(P_set[i_thread]);
            if (P_set_half[i_thread]) gsl_permutation_free(P_set_half[i_thread]);
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
    smooth_q = interpolatory_weights(n_col);
    smooth_q_half = interpolatory_weights(n_col / 2);
    make_dct_plan(n_col, dct_fft_size, dct_chirp, dct_kernel_fft);
    make_dct_plan(n_col / 2, dct_fft_size_half, dct_chirp_half, dct_kernel_fft_half);
    A_mat_set.assign(N_thread_max, std::vector<double>(d * d));
    const size_t dense_basis_size = use_banded_levin ? 0 : static_cast<size_t>(n_col) * n_col;
    const size_t dense_basis_half_size = use_banded_levin ? 0 : static_cast<size_t>(n_col / 2) * (n_col / 2);
    basis_vals_set.assign(N_thread_max, std::vector<double>(dense_basis_size));
    basis_prime_vals_set.assign(N_thread_max, std::vector<double>(dense_basis_size));
    basis_vals_half_set.assign(N_thread_max, std::vector<double>(dense_basis_half_size));
    basis_prime_vals_half_set.assign(N_thread_max, std::vector<double>(dense_basis_half_size));
    wA_set.assign(N_thread_max, std::vector<double>(d));
    wB_set.assign(N_thread_max, std::vector<double>(d));
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(N_thread_max > 1)
    for (uint i_thread = 0; i_thread < N_thread_max; i_thread++)
    {
        F_stacked_set[i_thread] = use_banded_levin ? nullptr : gsl_vector_alloc(d * n_col);
        F_stacked_set_half[i_thread] = use_banded_levin ? nullptr : gsl_vector_alloc(d * n_col / 2);
        ce_set[i_thread] = gsl_vector_alloc(d * n_col);
        ce_set_half[i_thread] = gsl_vector_alloc(d * n_col / 2);
        matrix_G_set[i_thread] = use_banded_levin ? nullptr : gsl_matrix_alloc(d * n_col, d * n_col);
        matrix_G_set_half[i_thread] = use_banded_levin ? nullptr : gsl_matrix_alloc(d * n_col / 2, d * n_col / 2);
        P_set[i_thread] = use_banded_levin ? nullptr : gsl_permutation_alloc(d * n_col);
        P_set_half[i_thread] = use_banded_levin ? nullptr : gsl_permutation_alloc(d * n_col / 2);
        if (F_stacked_set[i_thread]) gsl_vector_set_zero(F_stacked_set[i_thread]);
        if (F_stacked_set_half[i_thread]) gsl_vector_set_zero(F_stacked_set_half[i_thread]);
        setNodes_cheby(i_thread);
    }
}

void pylevin::init_splines(const array_double &x_in, const array_double &integrand_in, bool logx, bool logy)
{
    if (integrand_in.ndim() != 2 || integrand_in.shape(0) != x_in.size())
    {
        throw std::range_error("support dimensions must match integrand dimensions");
    }
    if (x_in.size() < 5)
    {
        throw std::range_error("the integrand needs at least 5 support points");
    }
    const auto integrand = integrand_in.unchecked<2>();
    if (!spline_coefficients.empty() && integrand.shape(1) != n_integrand)
    {
        throw std::range_error("If you update the integrand, they have to have the same shapes as in the constructor");
    }
    n_integrand = integrand.shape(1);
    is_x_log = logx;
    std::vector<double> x(x_in.data(), x_in.data() + x_in.size());
    for (uint i = 0; i < x.size(); i++)
    {
        if (is_x_log)
        {
            x[i] = log(x[i]);
        }
        if (i > 0 && !(x[i] > x[i - 1]))
        {
            throw std::range_error("x must be strictly increasing");
        }
    }
    if (x != x_integrand)
    {
        spline_moments.clear();
        grid_nodes.clear();
        sample_weights.clear();
        x_integrand = std::move(x);
        factor_spline_grid();
    }
    segment_cache.assign(N_thread_max, SegmentCache());
    spline_coefficients.resize(n_integrand);
    spline_samples.resize(n_integrand);
    const bool defer_linear = !sample_weights.empty();
    std::vector<char> log_y(n_integrand); // is_y_log packs bits, so it is not written in parallel
#pragma omp parallel num_threads(N_thread_max) if(n_integrand >= N_thread_max)
    {
        std::vector<double> secant(x_integrand.size() - 1), slope(x_integrand.size());
#pragma omp for schedule(static)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            std::vector<double> &y = spline_samples[i_integrand];
            y.resize(x_integrand.size());
            bool positive = true;
            for (uint i = 0; i < y.size(); i++)
            {
                y[i] = integrand(i, i_integrand);
                positive = positive && y[i] > 0;
            }
            log_y[i_integrand] = logy && positive;
            if (log_y[i_integrand])
            {
                for (double &value : y)
                {
                    value = log(value);
                }
            }
            if (log_y[i_integrand] || !defer_linear)
            {
                set_spline_coefficients(i_integrand, y, secant, slope);
            }
        }
    }
    is_y_log.assign(log_y.begin(), log_y.end());
    spline_coefficients_stale = defer_linear && std::find(log_y.begin(), log_y.end(), 0) != log_y.end();
}

void pylevin::ensure_spline_coefficients()
{
    if (!spline_coefficients_stale)
    {
        return;
    }
#pragma omp parallel num_threads(N_thread_max) if(n_integrand >= N_thread_max)
    {
        std::vector<double> secant(x_integrand.size() - 1), slope(x_integrand.size());
#pragma omp for schedule(static)
        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
        {
            if (!is_y_log[i_integrand])
            {
                set_spline_coefficients(i_integrand, spline_samples[i_integrand], secant, slope);
            }
        }
    }
    spline_coefficients_stale = false;
}

void pylevin::update_integrand(const array_double &x, const array_double &integrand, bool logx, bool logy)
{
    init_splines(x, integrand, logx, logy);
}

pylevin::array_double pylevin::get_integrand(const array_double &x_in)
{
    ensure_spline_coefficients();
    const double *x = x_in.data();
    array_double result_array({static_cast<size_t>(x_in.size()), static_cast<size_t>(n_integrand)});
    auto result = result_array.mutable_unchecked<2>();
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
    {
        for (uint i_x = 0; i_x < x_in.size(); i_x++)
        {
            result(i_x, i_integrand) = inhomogeneity(x[i_x], i_integrand, omp_get_thread_num());
        }
    }
    return result_array;
}

pybind11::tuple pylevin::get_quadrature()
{
    // The stored integrals as quadrature rules, sum_n weight_n f(x_n), with the nodes of
    // integral p at offsets[p]:offsets[p + 1] (p indexed like the stored bisection).
    if (!system_of_equations_set)
    {
        throw std::runtime_error("The quadrature rule is stored by the second integration call");
    }
    array_int offsets_array(bisection.size() + 1);
    auto offsets = offsets_array.mutable_unchecked<1>();
    offsets(0) = 0;
    for (uint p = 0; p < bisection.size(); p++)
    {
        size_t nodes = 0;
        for (const std::vector<double> &weights : weights_precomp[p])
        {
            nodes += weights.size();
        }
        offsets(p + 1) = offsets(p) + nodes;
    }
    array_double x_array(offsets(bisection.size()));
    array_double weights_array(offsets(bisection.size()));
    double *x = x_array.mutable_data();
    double *weights = weights_array.mutable_data();
    for (uint p = 0; p < bisection.size(); p++)
    {
        size_t node = offsets(p);
        for (uint i_bisec = 0; i_bisec + 1 < bisection[p].size(); i_bisec++)
        {
            for (uint j = 0; j < weights_precomp[p][i_bisec].size(); j++, node++)
            {
                x[node] = map_y_to_x(x_j_set[0][j], bisection[p][i_bisec], bisection[p][i_bisec + 1]);
                weights[node] = weights_precomp[p][i_bisec][j];
            }
        }
    }
    return pybind11::make_tuple(x_array, weights_array, offsets_array);
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
    if (type > 1)
    {
        return 0.0;
    }

    if (i == j)
    {
        const double inv_x = 1.0 / x;
        if (i == 0)
        {
            return static_cast<double>(ell) * inv_x;
        }
        const double offset = (type == 0) ? 2.0 : 1.0;
        return -(static_cast<double>(ell) + offset) * inv_x;
    }

    return (i < j) ? -k : k;
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
    if (!use_banded_levin)
    {
        for (uint j = 0; j < n_col; j++)
        {
            fill_chebyshev_basis(x_j_set[i][j], n_col,
                                 basis_vals_set[i].data() + j * n_col,
                                 basis_prime_vals_set[i].data() + j * n_col);
        }
        for (uint j = 0; j < n_col / 2; j++)
        {
            fill_chebyshev_basis(x_j_set_half[i][j], n_col / 2,
                                 basis_vals_half_set[i].data() + j * (n_col / 2),
                                 basis_prime_vals_half_set[i].data() + j * (n_col / 2));
        }
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
    if (!(x >= x_integrand.front() && x <= x_integrand.back()))
    {
        return 0.0;
    }
    uint &segment = segment_cache[tid].segment;
    segment = segment_of(x, segment);
    const std::array<double, 4> &c = spline_coefficients[i_integrand][segment];
    const double t = x - x_integrand[segment];
    const double result = c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    return is_y_log[i_integrand] ? exp(result) : result;
}

uint pylevin::segment_of(double x, uint guess) const
{
    // Segment s with x_s <= x < x_{s+1} (the last one for x at the upper end), trying guess first.
    if (x >= x_integrand[guess] && x < x_integrand[guess + 1])
    {
        return guess;
    }
    const uint upper = std::upper_bound(x_integrand.begin(), x_integrand.end(), x) - x_integrand.begin();
    return std::min<uint>(upper - 1, x_integrand.size() - 2);
}

void pylevin::ensure_dense_single_scratch(uint tid, uint col)
{
    const bool full_size = (col == n_col);
    gsl_vector *&rhs = full_size ? F_stacked_set[tid] : F_stacked_set_half[tid];
    gsl_matrix *&matrix = full_size ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_permutation *&permutation = full_size ? P_set[tid] : P_set_half[tid];
    std::vector<double> &basis = full_size ? basis_vals_set[tid] : basis_vals_half_set[tid];
    std::vector<double> &basis_prime = full_size ? basis_prime_vals_set[tid] : basis_prime_vals_half_set[tid];
    const std::vector<double> &nodes = full_size ? x_j_set[tid] : x_j_set_half[tid];
    if (!rhs)
    {
        rhs = gsl_vector_alloc(d * col);
        gsl_vector_set_zero(rhs);
    }
    if (!matrix) matrix = gsl_matrix_alloc(d * col, d * col);
    if (!permutation) permutation = gsl_permutation_alloc(d * col);
    if (basis.size() != static_cast<size_t>(col) * col)
    {
        basis.resize(static_cast<size_t>(col) * col);
        basis_prime.resize(static_cast<size_t>(col) * col);
        for (uint node = 0; node < col; ++node)
        {
            fill_chebyshev_basis(nodes[node], col, basis.data() + node * col,
                                 basis_prime.data() + node * col);
        }
    }
}

bool pylevin::solve_LSE_single_banded(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    // The polynomially banded formulation multiplies the single-Bessel Levin
    // system by x(1-y^2). Keep the existing collocation solver for intervals
    // which include the regular-singular point x=0 and for very small systems.
    if (type > 1 || col < 4 || col % 2 != 0 || A == 0.0 || B == 0.0 || A * B <= 0.0 || !std::isfinite(A) || !std::isfinite(B))
    {
        return false;
    }

    const uint components = d;
    const uint number_of_modes = col;
    const uint interior_modes = number_of_modes - 2;
    const uint reduced_size = components * interior_modes;
    const double half_BA = (B - A) / 2.0;
    const double center = (A + B) / 2.0;
    const uint tid = omp_get_thread_num();
    if (reduced_size == 0 || half_BA == 0.0)
    {
        return false;
    }

    const uint correction_count = 2 * components;
    auto evaluate_endpoint_operator = [&](const std::vector<double> &coefficients,
                                          bool at_right_endpoint, uint output_component)
    {
        const double x = at_right_endpoint ? B : A;
        double derivative = 0.0;
        for (uint mode = 0; mode < number_of_modes; ++mode)
        {
            const double coefficient = coefficients[output_component * number_of_modes + mode];
            const double derivative_sign = at_right_endpoint ? 1.0 : ((mode % 2 == 0) ? -1.0 : 1.0);
            derivative += derivative_sign * mode * mode * coefficient;
        }
        double system_value = 0.0;
        for (uint input_component = 0; input_component < components; ++input_component)
        {
            double p_value = 0.0;
            for (uint mode = 0; mode < number_of_modes; ++mode)
            {
                const double endpoint_sign = at_right_endpoint ? 1.0 : ((mode % 2 == 0) ? 1.0 : -1.0);
                p_value += endpoint_sign * coefficients[input_component * number_of_modes + mode];
            }
            system_value += A_matrix_single(input_component, output_component, x, k, ell) * p_value;
        }
        return derivative + half_BA * system_value;
    };

    auto double_bits = [](double value)
    {
        std::uint64_t bits;
        static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    const BandedCacheKey cache_key{double_bits(A), double_bits(B), double_bits(k),
                                   col, ell, type};
    std::shared_ptr<const BandedSystemCache> cached_system;
    {
        std::lock_guard<std::mutex> lock(banded_system_cache_mutex);
        const auto found = banded_system_cache.find(cache_key);
        if (found != banded_system_cache.end())
        {
            cached_system = found->second;
        }
    }

    if (!cached_system)
    {
        auto built = std::make_shared<BandedSystemCache>();
        // In y=(x-center)/half_BA coordinates the original equation is
        // p_y + half_BA*A(x)^T*p = half_BA*f. Multiplication by x(1-y^2)
        // removes the only rational coefficient and yields a banded operator.
        const std::vector<double> one_minus_y2{0.5, 0.0, -0.5};
        std::vector<std::vector<std::vector<double>>> zero_order_operator(
            components, std::vector<std::vector<double>>(components));
        for (uint output_component = 0; output_component < components; ++output_component)
        {
            for (uint input_component = 0; input_component < components; ++input_component)
            {
                std::vector<double> x_times_A_transpose;
                if (output_component == input_component)
                {
                    const double diagonal = (output_component == 0)
                                               ? static_cast<double>(ell)
                                               : -(static_cast<double>(ell) + ((type == 0) ? 2.0 : 1.0));
                    x_times_A_transpose = {half_BA * diagonal};
                }
                else
                {
                    const double sign = (input_component < output_component) ? -1.0 : 1.0;
                    x_times_A_transpose = {half_BA * sign * k * center,
                                           half_BA * sign * k * half_BA};
                }
                zero_order_operator[output_component][input_component] =
                    multiply_chebyshev_polynomials(one_minus_y2, x_times_A_transpose);
            }
        }

        constexpr int local_bandwidth = 3;
        constexpr size_t local_mode_count = 2 * local_bandwidth + 1;
        auto add_product_term = [&](std::array<double, 14> &local, uint output_component,
                                    uint polynomial_basis_mode, uint reference_mode,
                                    uint polynomial_mode, double coefficient)
        {
            if (coefficient == 0.0) return;
            const uint high_mode = alias_chebyshev_mode(polynomial_basis_mode + polynomial_mode,
                                                        number_of_modes);
            const uint low_degree = (polynomial_basis_mode > polynomial_mode)
                                        ? polynomial_basis_mode - polynomial_mode
                                        : polynomial_mode - polynomial_basis_mode;
            const uint low_mode = alias_chebyshev_mode(low_degree, number_of_modes);
            const int high_shift = static_cast<int>(high_mode) - static_cast<int>(reference_mode);
            const int low_shift = static_cast<int>(low_mode) - static_cast<int>(reference_mode);
            if (std::abs(high_shift) <= local_bandwidth)
                local[output_component * local_mode_count + (high_shift + local_bandwidth)] += 0.5 * coefficient;
            if (std::abs(low_shift) <= local_bandwidth)
                local[output_component * local_mode_count + (low_shift + local_bandwidth)] += 0.5 * coefficient;
        };
        auto apply_operator_to_basis = [&](uint input_component, uint input_mode)
        {
            std::array<double, 14> local{};
            for (uint output_component = 0; output_component < components; ++output_component)
            {
                const std::vector<double> &zero_order = zero_order_operator[output_component][input_component];
                for (uint polynomial_mode = 0; polynomial_mode < zero_order.size(); ++polynomial_mode)
                    add_product_term(local, output_component, input_mode, input_mode,
                                     polynomial_mode, zero_order[polynomial_mode]);
                if (output_component == input_component && input_mode > 0)
                {
                    const double scale = 0.5 * input_mode;
                    add_product_term(local, output_component, input_mode - 1, input_mode, 0, scale * center);
                    add_product_term(local, output_component, input_mode - 1, input_mode, 1, scale * half_BA);
                    add_product_term(local, output_component, input_mode + 1, input_mode, 0, -scale * center);
                    add_product_term(local, output_component, input_mode + 1, input_mode, 1, -scale * half_BA);
                }
            }
            return local;
        };

        const uint lower_bandwidth = components * local_bandwidth + components - 1;
        const uint upper_bandwidth = lower_bandwidth;
        const size_t matrix_stride = band_stride(lower_bandwidth, upper_bandwidth);
        std::vector<double> matrix(static_cast<size_t>(reduced_size) * matrix_stride, 0.0);
        std::vector<std::vector<double>> endpoint_columns(
            correction_count, std::vector<double>(reduced_size, 0.0));
        for (uint input_component = 0; input_component < components; ++input_component)
        {
            for (uint input_mode = 0; input_mode < number_of_modes; ++input_mode)
            {
                const auto column_values = apply_operator_to_basis(input_component, input_mode);
                const bool is_left_coefficient = input_mode == 0;
                const bool is_right_coefficient = input_mode == number_of_modes - 1;
                const uint correction_index = 2 * input_component + (is_right_coefficient ? 1 : 0);
                for (uint output_component = 0; output_component < components; ++output_component)
                {
                    for (int shift = -local_bandwidth; shift <= local_bandwidth; ++shift)
                    {
                        const double value = column_values[output_component * local_mode_count +
                                                           (shift + local_bandwidth)];
                        const int output_mode_signed = static_cast<int>(input_mode) + shift;
                        if (value == 0.0 || output_mode_signed < 1 ||
                            output_mode_signed >= static_cast<int>(number_of_modes - 1)) continue;
                        const uint output_mode = static_cast<uint>(output_mode_signed);
                        const uint row = (output_mode - 1) * components + output_component;
                        if (is_left_coefficient || is_right_coefficient)
                            endpoint_columns[correction_index][row] += value;
                        else
                        {
                            const uint matrix_column = (input_mode - 1) * components + input_component;
                            matrix[band_index(row, matrix_column, lower_bandwidth, matrix_stride)] += value;
                        }
                    }
                }
            }
        }

        built->original_matrix = matrix;
        built->factored_matrix = std::move(matrix);
        if (banded_lu_factor(built->factored_matrix, reduced_size,
                             lower_bandwidth, upper_bandwidth))
        {
            built->corrections.assign(correction_count,
                                      std::vector<double>(components * number_of_modes, 0.0));
            built->valid = true;
            for (uint correction = 0; correction < correction_count && built->valid; ++correction)
            {
                std::vector<double> correction_rhs(reduced_size, 0.0);
                for (uint row = 0; row < reduced_size; ++row)
                    correction_rhs[row] = -endpoint_columns[correction][row];
                const std::vector<double> original_correction_rhs = correction_rhs;
                if (!banded_lu_solve(built->factored_matrix, reduced_size, lower_bandwidth,
                                     upper_bandwidth, correction_rhs) ||
                    !banded_solve_residual_ok(built->original_matrix, reduced_size,
                                              lower_bandwidth, upper_bandwidth, correction_rhs,
                                              original_correction_rhs))
                {
                    built->valid = false;
                    break;
                }
                const uint component = correction / 2;
                const uint endpoint_mode = (correction % 2 == 0) ? 0 : number_of_modes - 1;
                built->corrections[correction][component * number_of_modes + endpoint_mode] = 1.0;
                for (uint mode = 1; mode + 1 < number_of_modes; ++mode)
                    for (uint output_component = 0; output_component < components; ++output_component)
                        built->corrections[correction][output_component * number_of_modes + mode] =
                            correction_rhs[(mode - 1) * components + output_component];
            }
            if (built->valid)
            {
                const uint boundary_size = 2 * components;
                built->boundary_matrix.assign(boundary_size * boundary_size, 0.0);
                for (uint endpoint = 0; endpoint < 2; ++endpoint)
                {
                    const bool at_right_endpoint = endpoint == 0;
                    for (uint output_component = 0; output_component < components; ++output_component)
                    {
                        const uint row = endpoint * components + output_component;
                        for (uint correction = 0; correction < correction_count; ++correction)
                            built->boundary_matrix[row * boundary_size + correction] =
                                evaluate_endpoint_operator(built->corrections[correction],
                                                           at_right_endpoint, output_component);
                    }
                }
                std::vector<double> matrix_check = built->boundary_matrix;
                std::vector<double> rhs_check(boundary_size, 0.0);
                built->valid = dense_pivot_solve(matrix_check, boundary_size, rhs_check);
            }
        }

        if (!built->valid)
        {
            // A failed system only needs its key and invalid flag for repeat fallback.
            std::vector<double>().swap(built->original_matrix);
            std::vector<double>().swap(built->factored_matrix);
            std::vector<std::vector<double>>().swap(built->corrections);
            std::vector<double>().swap(built->boundary_matrix);
        }
        auto cache_entry_bytes = [](const BandedSystemCache &entry)
        {
            size_t values = entry.original_matrix.size() + entry.factored_matrix.size() +
                            entry.boundary_matrix.size();
            for (const auto &correction : entry.corrections) values += correction.size();
            return values * sizeof(double);
        };
        constexpr size_t maximum_cached_systems = 8192;
        constexpr size_t maximum_cache_bytes = 64 * 1024 * 1024;
        if (cache_entry_bytes(*built) > maximum_cache_bytes)
        {
            cached_system = built;
        }
        else
        {
            std::lock_guard<std::mutex> lock(banded_system_cache_mutex);
            const auto inserted = banded_system_cache.emplace(cache_key, built);
            cached_system = inserted.first->second;
            if (inserted.second) banded_system_cache_bytes += cache_entry_bytes(*cached_system);
            while (banded_system_cache.size() > maximum_cached_systems ||
                   banded_system_cache_bytes > maximum_cache_bytes)
            {
                auto victim = banded_system_cache.begin();
                if (victim->first == cache_key && std::next(victim) != banded_system_cache.end()) ++victim;
                if (victim->first == cache_key) break;
                banded_system_cache_bytes -= cache_entry_bytes(*victim->second);
                banded_system_cache.erase(victim);
            }
        }
    }
    if (!cached_system || !cached_system->valid)
    {
        return false;
    }

    // The integrand-dependent transform and solves remain fresh on every call.
    std::vector<double> sampled_rhs(number_of_modes, 0.0);
    for (uint node = 1; node + 1 < number_of_modes; ++node)
    {
        const double y = std::cos(M_PI * node / (number_of_modes - 1));
        const double x = map_y_to_x(y, A, B);
        sampled_rhs[node] = (1.0 - y * y) * x * half_BA * inhomogeneity(x, i_integrand, tid);
    }
    std::vector<double> rhs_coefficients;
    const size_t convolution_size_for_col = (col == n_col) ? dct_fft_size : dct_fft_size_half;
    const std::vector<std::complex<double>> &chirp_for_col = (col == n_col) ? dct_chirp : dct_chirp_half;
    const std::vector<std::complex<double>> &kernel_fft_for_col =
        (col == n_col) ? dct_kernel_fft : dct_kernel_fft_half;
    chebyshev_coefficients_from_lobatto_values(sampled_rhs, rhs_coefficients,
                                                convolution_size_for_col, chirp_for_col,
                                                kernel_fft_for_col);
    std::vector<double> reduced_rhs(reduced_size, 0.0);
    for (uint mode = 1; mode + 1 < number_of_modes; ++mode)
        reduced_rhs[(mode - 1) * components] = rhs_coefficients[mode];
    const std::vector<double> original_rhs = reduced_rhs;
    const uint lower_bandwidth = components * 3 + components - 1;
    const uint upper_bandwidth = lower_bandwidth;
    if (!banded_lu_solve(cached_system->factored_matrix, reduced_size,
                         lower_bandwidth, upper_bandwidth, reduced_rhs) ||
        !banded_solve_residual_ok(cached_system->original_matrix, reduced_size,
                                  lower_bandwidth, upper_bandwidth, reduced_rhs, original_rhs))
    {
        return false;
    }
    std::vector<double> base_coefficients(components * number_of_modes, 0.0);
    for (uint mode = 1; mode + 1 < number_of_modes; ++mode)
        for (uint component = 0; component < components; ++component)
            base_coefficients[component * number_of_modes + mode] =
                reduced_rhs[(mode - 1) * components + component];

    const uint boundary_size = 2 * components;
    std::vector<double> boundary_rhs(boundary_size, 0.0);
    for (uint endpoint = 0; endpoint < 2; ++endpoint)
    {
        const bool at_right_endpoint = (endpoint == 0);
        const double x = at_right_endpoint ? B : A;
        const double source = half_BA * inhomogeneity(x, i_integrand, tid);
        for (uint output_component = 0; output_component < components; ++output_component)
        {
            const uint row = endpoint * components + output_component;
            boundary_rhs[row] = -evaluate_endpoint_operator(base_coefficients, at_right_endpoint, output_component);
            if (output_component == 0)
            {
                boundary_rhs[row] += source;
            }
        }
    }
    std::vector<double> boundary_matrix = cached_system->boundary_matrix;
    if (!dense_pivot_solve(boundary_matrix, boundary_size, boundary_rhs))
    {
        return false;
    }

    std::vector<double> coefficients = base_coefficients;
    for (uint correction = 0; correction < correction_count; ++correction)
    {
        for (uint index = 0; index < coefficients.size(); ++index)
        {
            coefficients[index] += boundary_rhs[correction] * cached_system->corrections[correction][index];
        }
    }
    gsl_vector *solution = (col == n_col) ? ce_set[tid] : ce_set_half[tid];
    for (uint index = 0; index < coefficients.size(); ++index)
    {
        gsl_vector_set(solution, index, coefficients[index]);
    }
    return true;
}

void pylevin::solve_LSE_single(double A, double B, uint col, uint i_integrand, double k, uint ell)
{
    // Pieces stored for repeat calls need the dense LU factorization for their weights.
    const bool store = bisection_set && !system_of_equations_set;
    if (use_banded_levin && !store && solve_LSE_single_banded(A, B, col, i_integrand, k, ell))
    {
        return;
    }
    uint tid = omp_get_thread_num();
    if (use_banded_levin)
    {
        ensure_dense_single_scratch(tid, col);
    }
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    const std::vector<double> &basis_vals = (col == n_col) ? basis_vals_set[tid] : basis_vals_half_set[tid];
    const std::vector<double> &basis_prime_vals = (col == n_col) ? basis_prime_vals_set[tid] : basis_prime_vals_half_set[tid];

    BandedCacheKey fallback_key{};
    std::shared_ptr<DenseFallbackCache> cached_fallback;
    if (use_banded_levin && !store)
    {
        fallback_key = BandedCacheKey{double_bit_pattern(A), double_bit_pattern(B),
                                      double_bit_pattern(k), col, ell, type};
        std::lock_guard<std::mutex> lock(dense_fallback_cache_mutex);
        const auto found = dense_fallback_cache.find(fallback_key);
        if (found != dense_fallback_cache.end())
        {
            cached_fallback = found->second;
            cached_fallback->last_used = ++dense_fallback_cache_clock;
        }
    }

    // A banded-system miss can fall back to the original dense collocation
    // system. Its operator is independent of the integrand, so reuse that LU
    // factorization while rebuilding only the right-hand side.
    if (cached_fallback)
    {
        for (uint j = 0; j < col; ++j)
        {
            const double x = map_y_to_x(x_j[j], A, B);
            gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        }
        gsl_linalg_LU_solve(cached_fallback->lu, cached_fallback->permutation,
                            F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
        return;
    }

    gsl_matrix_set_zero(matrix_G);
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        const uint basis_offset = j * col;
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_single(q, i, x, k, ell);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[basis_offset + m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[basis_offset + m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    if (use_banded_levin && !store)
    {
        auto built = std::make_shared<DenseFallbackCache>();
        const size_t dimension = static_cast<size_t>(d) * col;
        built->bytes = dimension * dimension * sizeof(double) +
                       dimension * sizeof(size_t);
        constexpr size_t maximum_dense_fallback_cache_bytes = 64 * 1024 * 1024;
        constexpr size_t maximum_dense_fallback_cache_entries = 8192;
        if (built->bytes <= maximum_dense_fallback_cache_bytes)
        {
            built->lu = gsl_matrix_alloc(dimension, dimension);
            built->permutation = gsl_permutation_alloc(dimension);
            gsl_matrix_memcpy(built->lu, matrix_G);
            gsl_permutation_memcpy(built->permutation, P);
            std::lock_guard<std::mutex> lock(dense_fallback_cache_mutex);
            const auto inserted = dense_fallback_cache.emplace(fallback_key, built);
            if (inserted.second)
            {
                built->last_used = ++dense_fallback_cache_clock;
                dense_fallback_cache_bytes += built->bytes;
                while (dense_fallback_cache.size() > maximum_dense_fallback_cache_entries ||
                       dense_fallback_cache_bytes > maximum_dense_fallback_cache_bytes)
                {
                    auto victim = dense_fallback_cache.end();
                    for (auto it = dense_fallback_cache.begin(); it != dense_fallback_cache.end(); ++it)
                    {
                        if (it->first != fallback_key &&
                            (victim == dense_fallback_cache.end() ||
                             it->second->last_used < victim->second->last_used))
                        {
                            victim = it;
                        }
                    }
                    if (victim == dense_fallback_cache.end()) break;
                    dense_fallback_cache_bytes -= victim->second->bytes;
                    dense_fallback_cache.erase(victim);
                }
            }
            else
            {
                inserted.first->second->last_used = ++dense_fallback_cache_clock;
            }
        }
    }
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
}

void pylevin::solve_LSE_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    uint tid = omp_get_thread_num();
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_matrix_set_zero(matrix_G);
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    const std::vector<double> &basis_vals = (col == n_col) ? basis_vals_set[tid] : basis_vals_half_set[tid];
    const std::vector<double> &basis_prime_vals = (col == n_col) ? basis_prime_vals_set[tid] : basis_prime_vals_half_set[tid];
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        const uint basis_offset = j * col;
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_double(q, i, x, k_1, k_2, ell_1, ell_2);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[basis_offset + m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[basis_offset + m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
}

void pylevin::solve_LSE_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    uint tid = omp_get_thread_num();
    gsl_matrix *matrix_G = (col == n_col) ? matrix_G_set[tid] : matrix_G_set_half[tid];
    gsl_matrix_set_zero(matrix_G);
    const double half_BA = (B - A) / 2.0;
    const std::vector<double> &x_j = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    gsl_vector *F_stacked = (col == n_col) ? F_stacked_set[tid] : F_stacked_set_half[tid];
    const std::vector<double> &basis_vals = (col == n_col) ? basis_vals_set[tid] : basis_vals_half_set[tid];
    const std::vector<double> &basis_prime_vals = (col == n_col) ? basis_prime_vals_set[tid] : basis_prime_vals_half_set[tid];
    std::vector<double> &A_mat = A_mat_set[tid];
    for (uint j = 0; j < col; j++)
    {
        double y = x_j[j];
        double x = map_y_to_x(y, A, B);
        gsl_vector_set(F_stacked, j, half_BA * inhomogeneity(x, i_integrand, tid));
        const uint basis_offset = j * col;
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
                A_mat[q * d + i] = A_matrix_triple(q, i, x, k_1, k_2, k_3, ell_1, ell_2, ell_3);
        for (uint i = 0; i < d; i++)
            for (uint q = 0; q < d; q++)
            {
                double A_qi = half_BA * A_mat[q * d + i];
                for (uint m = 0; m < col; m++)
                {
                    double LSE_coeff = A_qi * basis_vals[basis_offset + m];
                    if (q == i)
                        LSE_coeff += basis_prime_vals[basis_offset + m];
                    gsl_matrix_set(matrix_G, i * col + j, q * col + m, LSE_coeff);
                }
            }
    }
    int s;
    gsl_permutation *P = (col == n_col) ? P_set[tid] : P_set_half[tid];
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_linalg_LU_solve(matrix_G, P, F_stacked, (col == n_col) ? ce_set[tid] : ce_set_half[tid]);
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
    if (is_non_oscillatory(A, B, k))
    {
        return smooth_integral(A, B, col, i_integrand, [&](double x)
                               { return w_single_bessel(x, k, ell, 0); },
                               bisection_set && !system_of_equations_set);
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
    if (bisection_set && !system_of_equations_set)
    {
        store_levin_weights(A, B, i_integrand, wA, wB);
    }
    return result;
}

double pylevin::integrate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2)
{
    if (A == B)
    {
        return 0.;
    }
    if (is_non_oscillatory(A, B, k_1 + k_2))
    {
        return smooth_integral(A, B, col, i_integrand, [&](double x)
                               { return w_double_bessel(x, k_1, k_2, ell_1, ell_2, 0); },
                               bisection_set && !system_of_equations_set);
    }
    uint tid = omp_get_thread_num();
    double result = 0.0;
    solve_LSE_double(A, B, col, i_integrand, k_1, k_2, ell_1, ell_2);
    double *wA = wA_set[tid].data();
    double *wB = wB_set[tid].data();
    fill_w_double(A, k_1, k_2, ell_1, ell_2, wA);
    fill_w_double(B, k_1, k_2, ell_1, ell_2, wB);
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
    if (bisection_set && !system_of_equations_set)
    {
        store_levin_weights(A, B, i_integrand, wA, wB);
    }
    return result;
}

double pylevin::integrate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3)
{
    if (A == B)
    {
        return 0.;
    }
    if (is_non_oscillatory(A, B, k_1 + k_2 + k_3))
    {
        return smooth_integral(A, B, col, i_integrand, [&](double x)
                               { return w_triple_bessel(x, k_1, k_2, k_3, ell_1, ell_2, ell_3, 0); },
                               bisection_set && !system_of_equations_set);
    }
    uint tid = omp_get_thread_num();
    double result = 0.0;
    solve_LSE_triple(A, B, col, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3);
    double *wA = wA_set[tid].data();
    double *wB = wB_set[tid].data();
    fill_w_triple(A, k_1, k_2, k_3, ell_1, ell_2, ell_3, wA);
    fill_w_triple(B, k_1, k_2, k_3, ell_1, ell_2, ell_3, wB);
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
    if (bisection_set && !system_of_equations_set)
    {
        store_levin_weights(A, B, i_integrand, wA, wB);
    }
    return result;
}

double pylevin::integrate_lse_set(uint i_integrand)
{
    uint tid = omp_get_thread_num();
    const uint p = precomp_index(i_integrand, tid);
    double result = 0.0;
    if (!sample_weights.empty() && !is_y_log[i_integrand])
    {
        const SampleWeights &row = sample_weights[p];
        const double *y = spline_samples[i_integrand].data() + row.start;
        for (uint a = 0; a < row.weight.size(); a++)
        {
            result += row.weight[a] * y[a];
        }
        return result;
    }
    if (!spline_moments.empty() && !is_y_log[i_integrand])
    {
        const std::vector<std::array<double, 4>> &coefficients = spline_coefficients[i_integrand];
        for (const SplineMoment &entry : spline_moments[p])
        {
            const std::array<double, 4> &c = coefficients[entry.segment];
            result += entry.moment[0] * c[0] + entry.moment[1] * c[1] + entry.moment[2] * c[2] + entry.moment[3] * c[3];
        }
        return result;
    }
    if (!grid_nodes.empty() && is_y_log[i_integrand])
    {
        const std::vector<std::array<double, 4>> &coefficients = spline_coefficients[i_integrand];
        for (const GridNode &node : grid_nodes[p])
        {
            const std::array<double, 4> &c = coefficients[node.segment];
            result += node.weight * exp(c[0] + node.t * (c[1] + node.t * (c[2] + node.t * c[3])));
        }
        return result;
    }
    for (uint i_bisec = 0; i_bisec + 1 < bisection[p].size(); i_bisec++)
    {
        const double A = bisection[p][i_bisec];
        const double B = bisection[p][i_bisec + 1];
        const std::vector<double> &weights = weights_precomp[p][i_bisec];
        for (uint j = 0; j < weights.size(); j++)
        {
            result += weights[j] * inhomogeneity(map_y_to_x(x_j_set[tid][j], A, B), i_integrand, tid);
        }
    }
    return result;
}

uint pylevin::precomp_index(uint i_integrand, uint tid) const
{
    return is_diagonal ? index_variable[tid] : i_integrand * size_variables + index_variable[tid];
}

bool pylevin::integrand_resolved(double A, double B, uint i_integrand)
{
    // The integrand is resolved on [A, B] if the last two Chebyshev coefficients of its
    // interpolant on n_col Lobatto points are below tol_rel times the largest one.
    const uint tid = omp_get_thread_num();
    std::vector<double> values(n_col), coefficients;
    for (uint j = 0; j < n_col; j++)
    {
        values[j] = inhomogeneity(map_y_to_x(std::cos(M_PI * j / (n_col - 1)), A, B), i_integrand, tid);
    }
    chebyshev_coefficients_from_lobatto_values(values, coefficients, dct_fft_size, dct_chirp, dct_kernel_fft);
    double largest = 0.0;
    for (double coefficient : coefficients)
    {
        largest = std::max(largest, std::abs(coefficient));
    }
    return std::abs(coefficients[n_col - 1]) + std::abs(coefficients[n_col - 2]) <= tol_rel * largest;
}

bool pylevin::is_non_oscillatory(double A, double B, double k_sum) const
{
    return non_oscillatory > 0.0 && k_sum * (B - A) < non_oscillatory;
}

double pylevin::smooth_integral(double A, double B, uint col, uint i_integrand, const std::function<double(double)> &w0, bool store)
{
    const uint tid = omp_get_thread_num();
    const std::vector<double> &nodes = (col == n_col) ? x_j_set[tid] : x_j_set_half[tid];
    const std::vector<double> &q = (col == n_col) ? smooth_q : smooth_q_half;
    const double half_BA = (B - A) / 2.0;
    std::vector<double> *stored = nullptr;
    if (store)
    {
        stored = &weights_precomp[precomp_index(i_integrand, tid)][index_bisection[tid]];
        stored->assign(col, 0.0);
    }
    double result = 0.0;
    for (uint j = 0; j < col; j++)
    {
        const double x = map_y_to_x(nodes[j], A, B);
        const double weight = half_BA * q[j] * w0(x);
        if (stored)
        {
            (*stored)[j] = weight;
        }
        result += weight * inhomogeneity(x, i_integrand, tid);
    }
    return result;
}

void pylevin::store_levin_weights(double A, double B, uint i_integrand, const double *wA, const double *wB)
{
    uint tid = omp_get_thread_num();
    gsl_vector *v = ce_set[tid];
    for (uint i = 0; i < d; i++)
    {
        for (uint m = 0; m < n_col; m++)
        {
            gsl_vector_set(v, i * n_col + m, wB[i] - ((m % 2 == 0) ? wA[i] : -wA[i]));
        }
    }
    gsl_blas_dtrsv(CblasUpper, CblasTrans, CblasNonUnit, matrix_G_set[tid], v);
    gsl_blas_dtrsv(CblasLower, CblasTrans, CblasUnit, matrix_G_set[tid], v);
    gsl_permute_vector_inverse(P_set[tid], v);
    const double half_BA = (B - A) / 2.0;
    std::vector<double> &weights = weights_precomp[precomp_index(i_integrand, tid)][index_bisection[tid]];
    weights.resize(n_col);
    for (uint j = 0; j < n_col; j++)
    {
        weights[j] = half_BA * gsl_vector_get(v, j);
    }
}

void pylevin::build_grid_tables()
{
    const bool linear_y = std::find(is_y_log.begin(), is_y_log.end(), false) != is_y_log.end();
    const bool log_y = std::find(is_y_log.begin(), is_y_log.end(), true) != is_y_log.end();
    spline_moments.assign(linear_y ? bisection.size() : 0, std::vector<SplineMoment>());
    grid_nodes.assign(log_y ? bisection.size() : 0, std::vector<GridNode>());
#pragma omp parallel for num_threads(N_thread_max) schedule(auto)
    for (uint p = 0; p < bisection.size(); p++)
    {
        const std::vector<double> &nodes = x_j_set[omp_get_thread_num()];
        uint segment = 0;
        for (uint i_bisec = 0; i_bisec + 1 < bisection[p].size(); i_bisec++)
        {
            const std::vector<double> &weights = weights_precomp[p][i_bisec];
            for (uint j = 0; j < weights.size(); j++)
            {
                double x = map_y_to_x(nodes[j], bisection[p][i_bisec], bisection[p][i_bisec + 1]);
                if (is_x_log)
                {
                    x = log(x);
                }
                if (!(x >= x_integrand.front() && x <= x_integrand.back()))
                {
                    continue;
                }
                segment = segment_of(x, segment);
                const double t = x - x_integrand[segment];
                if (log_y)
                {
                    grid_nodes[p].push_back({segment, t, weights[j]});
                }
                if (linear_y)
                {
                    std::vector<SplineMoment> &moments = spline_moments[p];
                    if (moments.empty() || moments.back().segment != segment)
                    {
                        moments.push_back({segment, {0.0, 0.0, 0.0, 0.0}});
                    }
                    double term = weights[j];
                    for (double &moment : moments.back().moment)
                    {
                        moment += term;
                        term *= t;
                    }
                }
            }
        }
    }
    if (!linear_y)
    {
        sample_weights.clear();
        return;
    }
    sample_weights.assign(bisection.size(), SampleWeights());
#pragma omp parallel num_threads(N_thread_max)
    {
        std::vector<double> g_y(x_integrand.size()), g_s(x_integrand.size()), g_m(x_integrand.size() - 1);
#pragma omp for schedule(auto)
        for (uint p = 0; p < bisection.size(); p++)
        {
            fold_spline_moments(spline_moments[p], sample_weights[p], g_y, g_s, g_m);
        }
    }
    // Per repeat call, the moments need the spline coefficients (about 8 operations per sample
    // and linear-y integrand), the sample weights only the samples.
    size_t sample_total = 0, moment_total = 0;
    for (uint p = 0; p < bisection.size(); p++)
    {
        sample_total += sample_weights[p].weight.size();
        moment_total += 4 * spline_moments[p].size();
    }
    const size_t linear_integrands = std::count(is_y_log.begin(), is_y_log.end(), false);
    if (sample_total <= moment_total + 8 * linear_integrands * x_integrand.size())
    {
        spline_moments.clear();
    }
    else
    {
        sample_weights.clear();
    }
}

void pylevin::fold_spline_moments(const std::vector<SplineMoment> &moments, SampleWeights &row, std::vector<double> &g_y,
                                  std::vector<double> &g_s, std::vector<double> &g_m) const
{
    // sum_s moment_s . (y_s, b_s, c_s, d_s) is linear in the samples y; its weights g_y follow
    // from set_spline_coefficients run backwards (its adjoint). g_s and g_m collect the weights
    // of the knot slopes and secant slopes. Only samples near the touched segments are kept:
    // the influence of the others on the spline decays by a factor of at least ~0.27 per knot.
    const std::vector<double> &x = x_integrand;
    const std::vector<double> &h = spline_h;
    const std::vector<double> &inverse_h = spline_inverse_h;
    const uint n = x.size();
    row.weight.clear();
    if (moments.empty())
    {
        return;
    }
    std::fill(g_y.begin(), g_y.end(), 0.0);
    std::fill(g_s.begin(), g_s.end(), 0.0);
    std::fill(g_m.begin(), g_m.end(), 0.0);
    uint first = n, last = 0;
    for (const SplineMoment &entry : moments)
    {
        const uint i = entry.segment;
        const double ih = inverse_h[i];
        g_y[i] += entry.moment[0];
        g_s[i] += entry.moment[1] - 2.0 * ih * entry.moment[2] + ih * ih * entry.moment[3];
        g_s[i + 1] += -ih * entry.moment[2] + ih * ih * entry.moment[3];
        g_m[i] += 3.0 * ih * entry.moment[2] - 2.0 * ih * ih * entry.moment[3];
        first = std::min(first, i);
        last = std::max(last, i + 1);
    }
    g_s[0] *= spline_inverse_pivot[0];
    for (uint i = 1; i < n; i++)
    {
        g_s[i] = (g_s[i] - spline_upper[i - 1] * g_s[i - 1]) * spline_inverse_pivot[i];
    }
    for (uint i = n - 1; i-- > 0;)
    {
        g_s[i] -= spline_multiplier[i + 1] * g_s[i + 1];
    }
    const double start = x[2] - x[0];
    const double end = x[n - 1] - x[n - 3];
    g_m[0] += (h[0] + 2.0 * start) * h[1] / start * g_s[0];
    g_m[1] += h[0] * h[0] / start * g_s[0];
    for (uint i = 1; i + 1 < n; i++)
    {
        g_m[i - 1] += 3.0 * h[i] * g_s[i];
        g_m[i] += 3.0 * h[i - 1] * g_s[i];
    }
    g_m[n - 3] += h[n - 2] * h[n - 2] / end * g_s[n - 1];
    g_m[n - 2] += (2.0 * end + h[n - 2]) * h[n - 3] / end * g_s[n - 1];
    for (uint i = 0; i + 1 < n; i++)
    {
        g_y[i + 1] += inverse_h[i] * g_m[i];
        g_y[i] -= inverse_h[i] * g_m[i];
    }
    constexpr uint margin = 64;
    row.start = (first > margin) ? first - margin : 0;
    const uint stop = std::min(n, last + 1 + margin);
    row.weight.assign(g_y.begin() + row.start, g_y.begin() + stop);
}

void pylevin::factor_spline_grid()
{
    const std::vector<double> &x = x_integrand;
    const uint n = x.size();
    spline_h.resize(n - 1);
    spline_inverse_h.resize(n - 1);
    for (uint i = 0; i + 1 < n; i++)
    {
        spline_h[i] = x[i + 1] - x[i];
        spline_inverse_h[i] = 1.0 / spline_h[i];
    }
    std::vector<double> lower(n, 0.0), diagonal(n);
    spline_upper.assign(n, 0.0);
    diagonal[0] = spline_h[1];
    spline_upper[0] = x[2] - x[0];
    for (uint i = 1; i + 1 < n; i++)
    {
        lower[i] = spline_h[i];
        diagonal[i] = 2.0 * (spline_h[i - 1] + spline_h[i]);
        spline_upper[i] = spline_h[i - 1];
    }
    lower[n - 1] = x[n - 1] - x[n - 3];
    diagonal[n - 1] = spline_h[n - 3];
    spline_multiplier.assign(n, 0.0);
    spline_inverse_pivot.resize(n);
    spline_inverse_pivot[0] = 1.0 / diagonal[0];
    for (uint i = 1; i < n; i++)
    {
        spline_multiplier[i] = lower[i] * spline_inverse_pivot[i - 1];
        spline_inverse_pivot[i] = 1.0 / (diagonal[i] - spline_multiplier[i] * spline_upper[i - 1]);
    }
}

void pylevin::set_spline_coefficients(uint i_integrand, const std::vector<double> &y, std::vector<double> &m, std::vector<double> &slope)
{
    // Knot slopes s_i from the factorised system; each segment is then the cubic Hermite
    // polynomial through (y_i, s_i) and (y_{i+1}, s_{i+1}). m and slope are scratch.
    const std::vector<double> &x = x_integrand;
    const std::vector<double> &h = spline_h;
    const uint n = x.size();
    for (uint i = 0; i + 1 < n; i++)
    {
        m[i] = (y[i + 1] - y[i]) * spline_inverse_h[i];
    }
    const double start = x[2] - x[0];
    const double end = x[n - 1] - x[n - 3];
    slope[0] = ((h[0] + 2.0 * start) * h[1] * m[0] + h[0] * h[0] * m[1]) / start;
    for (uint i = 1; i + 1 < n; i++)
    {
        slope[i] = 3.0 * (h[i] * m[i - 1] + h[i - 1] * m[i]);
    }
    slope[n - 1] = (h[n - 2] * h[n - 2] * m[n - 3] + (2.0 * end + h[n - 2]) * h[n - 3] * m[n - 2]) / end;
    for (uint i = 1; i < n; i++)
    {
        slope[i] -= spline_multiplier[i] * slope[i - 1];
    }
    slope[n - 1] *= spline_inverse_pivot[n - 1];
    for (uint i = n - 1; i-- > 0;)
    {
        slope[i] = (slope[i] - spline_upper[i] * slope[i + 1]) * spline_inverse_pivot[i];
    }
    std::vector<std::array<double, 4>> &coefficients = spline_coefficients[i_integrand];
    coefficients.resize(n - 1);
    for (uint i = 0; i + 1 < n; i++)
    {
        const double inverse_h = spline_inverse_h[i];
        coefficients[i] = {y[i], slope[i], (3.0 * m[i] - 2.0 * slope[i] - slope[i + 1]) * inverse_h,
                           (slope[i] + slope[i + 1] - 2.0 * m[i]) * inverse_h * inverse_h};
    }
}

double pylevin::iterate_pieces(double A, double B, uint col, uint i_integrand, uint smax, bool verbose, double x_turn,
                               const std::function<double(double, double, uint)> &integrate, const std::string &label)
{
    uint tid = omp_get_thread_num();
    if (B - A < min_interval)
    {
        return 0.0;
    }
    std::vector<double> &stored_bisection = bisection[precomp_index(i_integrand, tid)];
    auto store_bisection = [&](const std::vector<double> &edges)
    {
        for (double edge : edges)
        {
            stored_bisection.push_back(edge);
        }
    };
    constexpr double forced_split_error = 1e300;
    auto too_wide = [&](double left, double right)
    {
        const double floor = std::max(left, x_turn);
        return max_ratio > 1.0 && floor > 0.0 && right / max_ratio > floor;
    };
    // A piece on which the integrand itself is not resolved counts as unconverged, even if
    // I_col and I_col/2 agree: both can miss small features such as the BAO wiggles of a
    // power spectrum.
    auto error_estimate = [&](double left, double right, double I_full, double I_half)
    {
        if (too_wide(left, right))
        {
            return forced_split_error;
        }
        const double estimate = pow(I_full - I_half, 2);
        return integrand_resolved(left, right, i_integrand) ? estimate : std::max(estimate, I_full * I_full);
    };
    std::vector<double> x_sub{A};
    if (too_wide(A, B))
    {
        const double start = std::max(A, x_turn);
        const uint lead = (A < start) ? 1 : 0;
        if (lead)
        {
            x_sub.push_back(start);
        }
        const double log_ratio = std::log(B / start);
        uint n_geometric = static_cast<uint>(std::floor(log_ratio / std::log(max_ratio))) + 1;
        n_geometric = std::min(n_geometric, smax + 2 - lead);
        for (uint q = 1; q < n_geometric; q++)
        {
            x_sub.push_back(start * std::exp(log_ratio * q / n_geometric));
        }
    }
    x_sub.push_back(B);
    const uint pieces = static_cast<uint>(x_sub.size()) - 1;
    std::vector<double> approximations, error_estimates;
    double result = 0.0;
    for (uint piece = 0; piece < pieces; piece++)
    {
        const double I_full = integrate(x_sub[piece], x_sub[piece + 1], col);
        const double I_half = (pieces == 1) ? 0.0 : integrate(x_sub[piece], x_sub[piece + 1], col / 2);
        approximations.push_back(I_full);
        error_estimates.push_back(error_estimate(x_sub[piece], x_sub[piece + 1], I_full, I_half));
        result += I_full;
    }
    double previous = 0.0;
    uint sub = pieces;
    while (sub <= smax + 1)
    {
        if (abs(result - previous) <= GSL_MAX(tol_rel * abs(result), tol_abs) && (sqrt(accumulate(error_estimates.begin(), error_estimates.end(), 0.0)) <= GSL_MAX(tol_rel * abs(result), tol_abs)))
        {
            store_bisection(x_sub);
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
                    std::cerr << "subintervals too narrow for further bisection for integrand " << i_integrand << label << std::endl;
                    store_bisection(x_sub);
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates[i - 1] = -1.0;
        }
        const double middle = too_wide(x_sub[i - 1], x_sub[i])
                                  ? ((x_sub[i - 1] < x_turn) ? x_turn : std::sqrt(x_sub[i - 1] * x_sub[i]))
                                  : (x_sub[i - 1] + x_sub[i]) / 2.;
        x_sub.insert(x_sub.begin() + i, middle);
        double x_subim1_i = (x_sub[i - 1]);
        double x_subi_i = (x_sub[i]);
        double x_subip1_i = (x_sub[i + 1]);
        double old_approx = approximations[i - 1];
        double I_half = integrate(x_subim1_i, x_subi_i, col / 2);
        double I_full = integrate(x_subim1_i, x_subi_i, col);
        approximations[i - 1] = I_full;
        error_estimates[i - 1] = error_estimate(x_subim1_i, x_subi_i, I_full, I_half);
        I_half = integrate(x_subi_i, x_subip1_i, col / 2);
        I_full = integrate(x_subi_i, x_subip1_i, col);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, error_estimate(x_subi_i, x_subip1_i, I_full, I_half));
        result = result - old_approx + approximations[i - 1] + I_full;
    }
    if (verbose)
    {
        std::cerr << "maximum number of bisections reached for integrand " << i_integrand << label << std::endl;
    }
    if (verbose)
    {
        std::cerr << "Convergence cannot be reached for the current settings for integrand " << i_integrand << " try to decrease the relative accuracy or increase the possible number of bisections or the number of collocation points." << std::endl;
    }
    store_bisection(x_sub);
    return result;
}

namespace
{
double turning_point(double k, uint ell)
{
    return (k > 0) ? (ell + 1.0) / k : std::numeric_limits<double>::max();
}
}

double pylevin::iterate_single(double A, double B, uint col, uint i_integrand, double k, uint ell, uint smax, bool verbose)
{
    std::ostringstream label;
    if (verbose)
    {
        label << " at k " << k << " and ell " << ell;
    }
    return iterate_pieces(A, B, col, i_integrand, smax, verbose, turning_point(k, ell),
                          [&](double a, double b, uint c)
                          { return integrate_single(a, b, c, i_integrand, k, ell); },
                          label.str());
}

double pylevin::iterate_double(double A, double B, uint col, uint i_integrand, double k_1, double k_2, uint ell_1, uint ell_2, uint smax, bool verbose)
{
    std::ostringstream label;
    if (verbose)
    {
        label << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2;
    }
    return iterate_pieces(A, B, col, i_integrand, smax, verbose,
                          std::min(turning_point(k_1, ell_1), turning_point(k_2, ell_2)),
                          [&](double a, double b, uint c)
                          { return integrate_double(a, b, c, i_integrand, k_1, k_2, ell_1, ell_2); },
                          label.str());
}

double pylevin::iterate_triple(double A, double B, uint col, uint i_integrand, double k_1, double k_2, double k_3, uint ell_1, uint ell_2, uint ell_3, uint smax, bool verbose)
{
    std::ostringstream label;
    if (verbose)
    {
        label << " at k_1 " << k_1 << " at k_2 " << k_2 << " and ell_1 " << ell_1 << " and ell_2 " << ell_2;
    }
    return iterate_pieces(A, B, col, i_integrand, smax, verbose,
                          std::min({turning_point(k_1, ell_1), turning_point(k_2, ell_2), turning_point(k_3, ell_3)}),
                          [&](double a, double b, uint c)
                          { return integrate_triple(a, b, c, i_integrand, k_1, k_2, k_3, ell_1, ell_2, ell_3); },
                          label.str());
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
    weights_precomp.assign(bisection.size(), std::vector<std::vector<double>>());
    for (uint i = 0; i < bisection.size(); i++)
    {
        weights_precomp[i].assign(bisection[i].size() - 1, std::vector<double>());
    }
}

void pylevin::levin_integrate_bessel_single(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_in, const array_int &ell_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in)
{
    size_variables = x_max_in.size();
    if (d > 2)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 0 or 1");
    }
    if (x_min_in.size() != x_max_in.size() || x_min_in.size() != k_in.size() || x_min_in.size() != ell_in.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    const double *x_min = x_min_in.data();
    const double *x_max = x_max_in.data();
    const double *k = k_in.data();
    const std::vector<uint> ell = bessel_orders(ell_in);
    const std::vector<uint> integrand_of = integrand_indices(integrand_in);
    if (!system_of_equations_set)
    {
        ensure_spline_coefficients();
    }
    if (system_of_equations_set)
    {
        if (!is_diagonal && size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable) = integrate_lse_set(integrand_of[i_variable]);
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
                bisection.resize(size_variables, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
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
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = levin_integrate_single_bessel(x_min[i_variable], x_max[i_variable], k[i_variable], ell[i_variable], integrand_of[i_variable]);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
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
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
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
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_single(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, integrand_of[i_variable], k[i_variable], ell[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
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
            build_grid_tables();
            system_of_equations_set = true;
        }
    }
}

void pylevin::levin_integrate_bessel_double(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_1_in, const array_double &k_2_in, const array_int &ell_1_in, const array_int &ell_2_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in)
{
    size_variables = x_max_in.size();
    if (d == 8 || d == 2)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 2 or 3");
    }
    if (x_min_in.size() != x_max_in.size() || x_min_in.size() != k_1_in.size() || x_min_in.size() != k_2_in.size() || x_min_in.size() != ell_1_in.size() || x_min_in.size() != ell_2_in.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    const double *x_min = x_min_in.data();
    const double *x_max = x_max_in.data();
    const double *k_1 = k_1_in.data();
    const double *k_2 = k_2_in.data();
    const std::vector<uint> ell_1 = bessel_orders(ell_1_in);
    const std::vector<uint> ell_2 = bessel_orders(ell_2_in);
    const std::vector<uint> integrand_of = integrand_indices(integrand_in);
    if (!system_of_equations_set)
    {
        ensure_spline_coefficients();
    }
    if (system_of_equations_set)
    {
        if (!is_diagonal && size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable) = integrate_lse_set(integrand_of[i_variable]);
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
                bisection.resize(size_variables, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
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
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = levin_integrate_double_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable], integrand_of[i_variable]);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
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
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
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
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_double(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, integrand_of[i_variable], k_1[i_variable], k_2[i_variable], ell_1[i_variable], ell_2[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
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
            build_grid_tables();
            system_of_equations_set = true;
        }
    }
}

void pylevin::levin_integrate_bessel_triple(const array_double &x_min_in, const array_double &x_max_in, const array_double &k_1_in, const array_double &k_2_in, const array_double &k_3_in, const array_int &ell_1_in, const array_int &ell_2_in, const array_int &ell_3_in, pybind11::array_t<double> &result, const std::optional<array_int> &integrand_in)
{
    size_variables = x_max_in.size();
    if (d != 8)
    {
        throw std::range_error("You have chosen the wrong integral type to call this function, must be either 4 or 5");
    }
    if (x_min_in.size() != x_max_in.size() || x_min_in.size() != k_1_in.size() || x_min_in.size() != k_2_in.size() || x_min_in.size() != ell_1_in.size() || x_min_in.size() != ell_2_in.size() || x_min_in.size() != ell_3_in.size() || x_min_in.size() != k_3_in.size())
    {
        throw std::range_error("sizes of all arguments must match");
    }
    const double *x_min = x_min_in.data();
    const double *x_max = x_max_in.data();
    const double *k_1 = k_1_in.data();
    const double *k_2 = k_2_in.data();
    const double *k_3 = k_3_in.data();
    const std::vector<uint> ell_1 = bessel_orders(ell_1_in);
    const std::vector<uint> ell_2 = bessel_orders(ell_2_in);
    const std::vector<uint> ell_3 = bessel_orders(ell_3_in);
    const std::vector<uint> integrand_of = integrand_indices(integrand_in);
    if (!system_of_equations_set)
    {
        ensure_spline_coefficients();
    }
    if (system_of_equations_set)
    {
        if (!is_diagonal && size_variables < n_integrand)
        {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
            for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
            {
                uint tid = omp_get_thread_num();
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                }
            }
        }
        else
        {
            if (!is_diagonal)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                    {
                        result.mutable_at(i_variable, i_integrand) = integrate_lse_set(i_integrand);
                    }
                }
            }
            else
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                {
                    uint tid = omp_get_thread_num();
                    index_variable[tid] = i_variable;
                    result.mutable_at(i_variable) = integrate_lse_set(integrand_of[i_variable]);
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
                bisection.resize(size_variables, std::vector<double>());
            }
            else
            {
                bisection.resize(n_integrand * size_variables, std::vector<double>());
            }
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
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
                        result.mutable_at(i_variable) = levin_integrate_triple_bessel(x_min[i_variable], x_max[i_variable], k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable], integrand_of[i_variable]);
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
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
            if (!is_diagonal && size_variables < n_integrand)
            {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(n_integrand >= N_thread_max)
                for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                {
                    uint tid = omp_get_thread_num();
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
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        result.mutable_at(i_variable) = 0.;
                        for (uint i_bisec = 0; i_bisec < bisection[i_variable].size() - 1; i_bisec++)
                        {
                            index_bisection[tid] = i_bisec;
                            result.mutable_at(i_variable) += integrate_triple(bisection[i_variable][i_bisec], bisection[i_variable][i_bisec + 1], n_col, integrand_of[i_variable], k_1[i_variable], k_2[i_variable], k_3[i_variable], ell_1[i_variable], ell_2[i_variable], ell_3[i_variable]);
                        }
                    }
                }
                else
                {
#pragma omp parallel for num_threads(N_thread_max) schedule(auto) if(size_variables >= N_thread_max)
                    for (uint i_variable = 0; i_variable < size_variables; i_variable++)
                    {
                        uint tid = omp_get_thread_num();
                        index_variable[tid] = i_variable;
                        for (uint i_integrand = 0; i_integrand < n_integrand; i_integrand++)
                        {
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
            build_grid_tables();
            system_of_equations_set = true;
        }
    }
}

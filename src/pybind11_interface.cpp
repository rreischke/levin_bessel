#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "pylevin.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(pylevin, m)
{
     m.doc() = ("Compute integrals with levin's method of products of up to three Bessel function of the first kind"
                 "Constructor of the pylevin class\n"
              "\n"
              "Initialises the Levin integrator class given the type of Bessel function"
              "and if it is a product or not, reads in the integrands and their support"
              "and wether they should be interpolated logarithmically or not.");

     py::class_<pylevin>(m, "pylevin")
         .def(py::init<uint, const pylevin::array_double &, const pylevin::array_double &, bool, bool, int, bool>(),
              "type"_a, "x"_a, "integrand"_a, "logx"_a, "logy"_a, "nthread"_a, py::arg("diagonal") = false)
         .def("set_levin", &pylevin::set_levin,
              "n_col_in"_a, "maximum_number_bisections_in"_a, "relative_accuracy_in"_a, "super_accurate"_a, "verbose"_a,
              py::arg("tol_abs_in") = 0.0,
              py::arg("banded") = true,
              py::arg("max_ratio") = 5.0,
              py::arg("non_oscillatory") = 0.5,
              py::call_guard<py::gil_scoped_release>(), R"mydelimiter(
    Select the banded Chebyshev Levin solver for single-Bessel integrals
    (types 0 and 1, enabled by default for at least 64 coefficients).
    Smaller systems, unsupported intervals, and failed factorizations use
    the existing dense collocation solver.

    max_ratio caps b / max(a, x_t) of every subinterval, with
    x_t = min_i (ell_i + 1) / k_i roughly where the Bessel functions start to
    oscillate. Without it, the error estimate can report convergence on
    subintervals spanning many decades in x while the result is off by up to
    ~1e-2. Wide intervals are split into such pieces from the start; the
    pieces count towards the maximum number of bisections. 0 disables it.

    non_oscillatory: pieces with (k_1 + k_2 + ...)(b - a) below this value
    use interpolatory quadrature on the collocation nodes instead of Levin.
    There, Levin's collocation matrix is nearly singular (results good to
    ~1e-4, or off by O(1) for ell = 0 at small k). 0 disables it.

    max_ratio=0 and non_oscillatory=0 reproduce the previous behaviour.

    Parameters
    ----------
)mydelimiter")
         .def("get_integrand", &pylevin::get_integrand,
              "x"_a)
         .def("get_quadrature", &pylevin::get_quadrature)
         .def("update_integrand", &pylevin::update_integrand,
              "x"_a, "integrand"_a, "logx"_a, "logy"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_single", &pylevin::levin_integrate_bessel_single,
              "x_min"_a, "x_max"_a, "k"_a, "ell"_a, "result"_a, py::arg("integrand") = py::none(),
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_double", &pylevin::levin_integrate_bessel_double,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "ell_1"_a, "ell_2"_a, "result"_a, py::arg("integrand") = py::none(),
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_triple", &pylevin::levin_integrate_bessel_triple,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "k_3"_a, "ell_1"_a, "ell_2"_a, "ell_3"_a, "result"_a, py::arg("integrand") = py::none(),
              py::call_guard<py::gil_scoped_release>());
}

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
         .def(py::init<uint, std::vector<double>, std::vector<std::vector<double>>, bool, bool, int, bool>(),
              "type"_a, "x"_a, "integrand"_a, "logx"_a, "logy"_a, "nthread"_a, py::arg("diagonal") = false)
         .def("set_levin", &pylevin::set_levin,
              "n_col_in"_a, "maximum_number_bisections_in"_a, "relative_accuracy_in"_a, "super_accurate"_a, "verbose"_a,
              py::call_guard<py::gil_scoped_release>(), R"mydelimiter(
    The foo function

    Parameters
    ----------
)mydelimiter")
         .def("get_integrand", &pylevin::get_integrand,
              "x"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("update_integrand", &pylevin::update_integrand,
              "x"_a, "integrand"_a, "logx"_a, "logy"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_single", &pylevin::levin_integrate_bessel_single,
              "x_min"_a, "x_max"_a, "k"_a, "ell"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_double", &pylevin::levin_integrate_bessel_double,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "ell_1"_a, "ell_2"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_triple", &pylevin::levin_integrate_bessel_triple,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "k_3"_a, "ell_1"_a, "ell_2"_a, "ell_3"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>());
}

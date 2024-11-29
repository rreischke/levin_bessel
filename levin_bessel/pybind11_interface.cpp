#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include "levin.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(levin, m)
{
     m.doc() = "Compute integrals with levin's method.";

     py::class_<levin>(m, "levin")
         .def(py::init<uint, std::vector<double>, std::vector<std::vector<double>>, bool, bool, int>(),
              "type"_a, "x"_a, "integrand"_a, "logx"_a, "logy"_a, "nthread"_a)
          """
          Initialises the Levin integrator class given the type of Bessel function 
          and if it is a product or not, reads in the integrands and their support
          and wether they should be interpolated logarithmically or not.

          Args:
          ----------
               type (integer): Specifies the type of integral you want to calculate.
                               Currently, the following types are implemented:

                                   * 0: single spherical Bessel function
                                   * 1: single cylindrical Bessel function
                                   * 2: double spherical Bessel function
                                   * 3: double cylindrical Bessel function
                                   * 4: triple spherical Bessel function
                                   * 5: triple cylindrical Bessel function
               
               x    (1d array):    One dimensional array over the support of the integrand :math:`x`.
                                   The integration used later when integrating should be within the
                                   minimum and maximum value of this array.

               integrand (2d array) Two dimensional array of the integrands :math:`f(x)`. Needs at least
                                    the shape ``len(x), 1``. Generally it should have the shape ``len(x), N``,
                                    where ``N``is the number of different integrals you want to calculate.
                                    In other words, you can pass a series of different :math: `f(x)`.
                                    
          """
          
         .def("set_levin", &levin::set_levin,
              "n_col_in"_a, "maximum_number_bisections_in"_a, "relative_accuracy_in"_a, "super_accurate"_a, "verbose"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_integrand", &levin::get_integrand,
              "x"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("update_integrand", &levin::update_integrand,
              "x"_a, "integrand"_a, "logx"_a, "logy"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_bisection", &levin::get_bisection,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_single", &levin::levin_integrate_bessel_single,
              "x_min"_a, "x_max"_a, "k"_a, "ell"_a, "diagonal"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_double", &levin::levin_integrate_bessel_double,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "ell_1"_a, "ell_2"_a, "diagonal"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("levin_integrate_bessel_triple", &levin::levin_integrate_bessel_triple,
              "x_min"_a, "x_max"_a, "k_1"_a, "k_2"_a, "k_3"_a, "ell_1"_a, "ell_2"_a, "ell_3"_a, "diagonal"_a, "result"_a,
              py::call_guard<py::gil_scoped_release>());
}

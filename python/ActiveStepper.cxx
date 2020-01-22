// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/ActiveStepper.hpp>
#include <Optima/Options.hpp>
#include <Optima/SaddlePointMatrix.hpp>
using namespace Optima;

void exportActiveStepper(py::module& m)
{
    py::class_<ActiveStepperProblem>(m, "ActiveStepperProblem")
        .def(py::init<
            VectorConstRef,  // x
            VectorConstRef,  // y
            MatrixConstRef,  // A
            VectorConstRef,  // b
            VectorConstRef,  // h
            MatrixConstRef,  // J
            VectorConstRef,  // g
            MatrixConstRef,  // H
            VectorConstRef,  // xlower
            VectorConstRef,  // xupper
            IndicesConstRef, // ilower
            IndicesConstRef, // iupper
            IndicesConstRef  // ifixed
        >())
        .def_readonly("x", &ActiveStepperProblem::x, "The current state of the primal variables of the canonical optimization problem.")
        .def_readonly("y", &ActiveStepperProblem::y, "The current state of the Lagrange multipliers of the canonical optimization problem.")
        .def_readonly("A", &ActiveStepperProblem::A, "The coefficient matrix of the linear equality constraints of the canonical optimization problem.")
        .def_readonly("b", &ActiveStepperProblem::b, "The right-hand side vector of the linear equality constraints of the canonical optimization problem.")
        .def_readonly("h", &ActiveStepperProblem::h, "The value of the equality constraint function.")
        .def_readonly("J", &ActiveStepperProblem::J, "The Jacobian of the equality constraint function.")
        .def_readonly("g", &ActiveStepperProblem::g, "The gradient of the objective function.")
        .def_readonly("H", &ActiveStepperProblem::H, "The Hessian of the objective function.")
        .def_readonly("xlower", &ActiveStepperProblem::xlower, "The values of the lower bounds of the variables constrained with lower bounds.")
        .def_readonly("xupper", &ActiveStepperProblem::xupper, "The values of the upper bounds of the variables constrained with upper bounds.")
        .def_readonly("ilower", &ActiveStepperProblem::ilower, "The indices of the variables with lower bounds.")
        .def_readonly("iupper", &ActiveStepperProblem::iupper, "The indices of the variables with upper bounds.")
        .def_readonly("ifixed", &ActiveStepperProblem::ifixed, "The indices of the variables with fixed values.")
        ;

    py::class_<ActiveStepper>(m, "ActiveStepper")
        .def(py::init<>())
        .def("setOptions", &ActiveStepper::setOptions)
        .def("decompose", &ActiveStepper::decompose)
        .def("solve", &ActiveStepper::solve)
        .def("step", &ActiveStepper::step)
        .def("residual", &ActiveStepper::residual)
        .def("matrix", &ActiveStepper::matrix)
        ;
}

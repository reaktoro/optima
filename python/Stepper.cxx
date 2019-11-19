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
#include <Optima/Constraints.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/Options.hpp>
#include <Optima/Stepper.hpp>
#include <Optima/Result.hpp>
using namespace Optima;

void exportStepper(py::module& m)
{
    py::class_<StepperProblem>(m, "StepperProblem")
        .def(py::init<
            VectorConstRef,  // x
            VectorConstRef,  // y
            VectorConstRef,  // z
            VectorConstRef,  // w
            VectorConstRef,  // xlower
            VectorConstRef,  // xupper
            VectorConstRef,  // b
            VectorConstRef,  // g
            MatrixConstRef   // H
        >())
        .def_readonly("x", &StepperProblem::x, "The current state of the primal variables of the canonical optimization problem.")
        .def_readonly("y", &StepperProblem::y, "The current state of the Lagrange multipliers of the canonical optimization problem.")
        .def_readonly("z", &StepperProblem::z, "The current state of the complementarity variables of the lower bounds of the canonical optimization problem.")
        .def_readonly("w", &StepperProblem::w, "The current state of the complementarity variables of the upper bounds of the canonical optimization problem.")
        .def_readonly("xlower", &StepperProblem::xlower, "The lower bound values of the canonical optimization problem.")
        .def_readonly("xupper", &StepperProblem::xupper, "The upper bound values of the canonical optimization problem.")
        .def_readonly("b", &StepperProblem::b, "The right-hand side vector of the linear equality constraints of the canonical optimization problem.")
        .def_readonly("g", &StepperProblem::g, "The gradient of the objective function.")
        .def_readonly("H", &StepperProblem::H, "The Hessian of the objective function.")
        ;

    py::class_<Stepper>(m, "Stepper")
        .def(py::init<>())
        .def(py::init<const Constraints&>())
        .def("setOptions", &Stepper::setOptions)
        .def("decompose", &Stepper::decompose)
        .def("solve", &Stepper::solve)
        .def("step", &Stepper::step)
        .def("residual", &Stepper::residual)
        .def("matrix", &Stepper::matrix)
        ;
}

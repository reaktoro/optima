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
    // py::class_<StepperProblem>(m, "StepperProblem")
    //     .def(py::init<
    //         VectorConstRef,  // x
    //         VectorConstRef,  // y
    //         VectorConstRef,  // z
    //         VectorConstRef,  // w
    //         MatrixConstRef,  // A
    //         VectorConstRef,  // b
    //         VectorConstRef,  // h
    //         MatrixConstRef,  // J
    //         VectorConstRef,  // g
    //         MatrixConstRef,  // H
    //         VectorConstRef,  // xlower
    //         VectorConstRef,  // xupper
    //         IndicesConstRef, // ilower
    //         IndicesConstRef, // iupper
    //         IndicesConstRef  // ifixed
    //     >())
    //     .def_readonly("x", &StepperProblem::x, "The current state of the primal variables of the canonical optimization problem.")
    //     .def_readonly("y", &StepperProblem::y, "The current state of the Lagrange multipliers of the canonical optimization problem.")
    //     .def_readonly("z", &StepperProblem::z, "The current state of the complementarity variables of the lower bounds of the canonical optimization problem.")
    //     .def_readonly("w", &StepperProblem::w, "The current state of the complementarity variables of the upper bounds of the canonical optimization problem.")
    //     .def_readonly("A", &StepperProblem::A, "The coefficient matrix of the linear equality constraints of the canonical optimization problem.")
    //     .def_readonly("b", &StepperProblem::b, "The right-hand side vector of the linear equality constraints of the canonical optimization problem.")
    //     .def_readonly("h", &StepperProblem::h, "The value of the equality constraint function.")
    //     .def_readonly("J", &StepperProblem::J, "The Jacobian of the equality constraint function.")
    //     .def_readonly("g", &StepperProblem::g, "The gradient of the objective function.")
    //     .def_readonly("H", &StepperProblem::H, "The Hessian of the objective function.")
    //     .def_readonly("xlower", &StepperProblem::xlower, "The values of the lower bounds of the variables constrained with lower bounds.")
    //     .def_readonly("xupper", &StepperProblem::xupper, "The values of the upper bounds of the variables constrained with upper bounds.")
    //     .def_readonly("ilower", &StepperProblem::ilower, "The indices of the variables with lower bounds.")
    //     .def_readonly("iupper", &StepperProblem::iupper, "The indices of the variables with upper bounds.")
    //     .def_readonly("ifixed", &StepperProblem::ifixed, "The indices of the variables with fixed values.")
    //     ;

    // py::class_<Stepper>(m, "Stepper")
    //     .def(py::init<>())
    //     .def("setOptions", &Stepper::setOptions)
    //     .def("decompose", &Stepper::decompose)
    //     .def("solve", &Stepper::solve)
    //     .def("step", &Stepper::step)
    //     .def("residual", &Stepper::residual)
    //     .def("matrix", &Stepper::matrix)
    //     ;
}

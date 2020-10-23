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
#include <Optima/JacobianMatrix.hpp>
#include <Optima/NewtonStepSolver.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/SolutionVector.hpp>
using namespace Optima;

void exportNewtonStepSolver(py::module& m)
{
    py::class_<NewtonStepSolver>(m, "NewtonStepSolver")
        .def(py::init<Index, Index, Index, Index>())
        .def("setOptions", &NewtonStepSolver::setOptions)
        .def("options", &NewtonStepSolver::options)
        .def("decompose", &NewtonStepSolver::decompose)
        .def("compute", &NewtonStepSolver::compute)
        ;
}

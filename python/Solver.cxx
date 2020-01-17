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
#include <Optima/Options.hpp>
#include <Optima/Params.hpp>
#include <Optima/Result.hpp>
#include <Optima/Solver.hpp>
#include <Optima/State.hpp>
using namespace Optima;

void exportSolver(py::module& m)
{
    // using ObjectivePtr = std::function<void(VectorConstRef, ObjectiveResult*)>;

	// // This is a workaround to let Python callback change the state of ObjectiveResult, and not a copy
    // auto createSolver = [](const ObjectivePtr& pyobjective, const Constraints& constraints)
    // {
    //     ObjectiveFunction objective = [=](VectorConstRef x, ObjectiveResult& f) { pyobjective(x, &f); };
    //     return Solver(objective, constraints);
    // };

    // py::class_<Solver>(m, "Solver")
    //     .def(py::init<>())
    //     .def(py::init(createSolver))
    //     .def("setOptions", &Solver::setOptions)
    //     .def("solve", &Solver::solve)
    //     .def("dxdp", &Solver::dxdp)
    //     ;
}

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
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/ObjectiveFunction.hpp>
using namespace Optima;

void exportObjectiveFunction(py::module& m)
{
    auto get_fxx = [](const ObjectiveResult& s) { return s.fxx; };
    auto get_fxp = [](const ObjectiveResult& s) { return s.fxp; };
    auto set_fxx = [](ObjectiveResult& s, MatrixView4py fxx) { s.fxx = fxx; };
    auto set_fxp = [](ObjectiveResult& s, MatrixView4py fxp) { s.fxp = fxp; };

    py::class_<ObjectiveResult>(m, "ObjectiveResult")
        .def_readwrite("f", &ObjectiveResult::f)
        .def_readwrite("fx", &ObjectiveResult::fx)
        .def_property("fxx", get_fxx, set_fxx)
        .def_property("fxp", get_fxp, set_fxp)
        .def_readwrite("diagfxx", &ObjectiveResult::diagfxx)
        .def_readwrite("fxx4basicvars", &ObjectiveResult::fxx4basicvars)
        .def_readwrite("succeeded", &ObjectiveResult::succeeded)
        ;

    py::class_<ObjectiveOptions::Eval>(m, "ObjectiveOptionsEval")
        .def_readwrite("fxx", &ObjectiveOptions::Eval::fxx, "True if evaluating the Jacobian matrix fxx is needed.")
        .def_readwrite("fxp", &ObjectiveOptions::Eval::fxp, "True if evaluating the Jacobian matrix fxp is needed.")
        ;

    py::class_<ObjectiveOptions>(m, "ObjectiveOptions")
        .def_readonly("eval", &ObjectiveOptions::eval, "The objective function components that need to be evaluated.")
        .def_readonly("ibasicvars", &ObjectiveOptions::ibasicvars, "The indices of the basic variables in x.")
        ;

    py::class_<ObjectiveFunction>(m, "ObjectiveFunction")
        .def(py::init<const ObjectiveFunction::Signature4py&>())
        .def("__call__", &ObjectiveFunction::operator())
        ;

    py::implicitly_convertible<ObjectiveFunction::Signature4py, ObjectiveFunction>();
}

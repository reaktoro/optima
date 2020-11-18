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
#include <Optima/ConstraintFunction.hpp>
using namespace Optima;

void exportConstraintFunction(py::module& m)
{
    auto get_ddx = [](const ConstraintResult& s) { return s.ddx; };
    auto get_ddp = [](const ConstraintResult& s) { return s.ddp; };
    auto set_ddx = [](ConstraintResult& s, MatrixConstRef4py ddx) { s.ddx = ddx; };
    auto set_ddp = [](ConstraintResult& s, MatrixConstRef4py ddp) { s.ddp = ddp; };

    py::class_<ConstraintResult>(m, "ConstraintResult")
        .def_readwrite("val", &ConstraintResult::val, "The evaluated vector value of c(x, p).")
        .def_property("ddx", get_ddx, set_ddx, "The evaluated Jacobian matrix of c(x, p) with respect to x.")
        .def_property("ddp", get_ddp, set_ddp, "The evaluated Jacobian matrix of c(x, p) with respect to p.")
        .def_readwrite("ddx4basicvars", &ConstraintResult::ddx4basicvars, "True if `ddx` is non-zero only on columns corresponding to basic varibles in x")
        ;

    py::class_<ConstraintOptions::Eval>(m, "ConstraintOptionsEval")
        .def_readwrite("ddx", &ConstraintOptions::Eval::ddx, "True if evaluating the Jacobian matrix of c(x, p) with respect to x is needed")
        .def_readwrite("ddp", &ConstraintOptions::Eval::ddp, "True if evaluating the Jacobian matrix of c(x, p) with respect to p is needed")
        ;

    py::class_<ConstraintOptions>(m, "ConstraintOptions")
        .def_readonly("eval", &ConstraintOptions::eval, "The objective function components that need to be evaluated.")
        .def_readonly("ibasicvars", &ConstraintOptions::ibasicvars, "The indices of the basic variables in x.")
        ;
}

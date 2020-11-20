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
    auto get_val = [](ConstraintResult& s) -> VectorRef { return s.val; };
    auto get_ddx = [](ConstraintResult& s) -> MatrixRef { return s.ddx; };
    auto get_ddp = [](ConstraintResult& s) -> MatrixRef { return s.ddp; };

    auto set_val = [](ConstraintResult& s, VectorView val) { s.val = val; };
    auto set_ddx = [](ConstraintResult& s, MatrixView4py ddx) { s.ddx = ddx; };
    auto set_ddp = [](ConstraintResult& s, MatrixView4py ddp) { s.ddp = ddp; };

    py::class_<ConstraintResult>(m, "ConstraintResult")
        .def_property("val", get_val, set_val)
        .def_property("ddx", get_ddx, set_ddx)
        .def_property("ddp", get_ddp, set_ddp)
        .def_readwrite("ddx4basicvars", &ConstraintResult::ddx4basicvars)
        .def_readwrite("succeeded", &ConstraintResult::succeeded)
        ;

    py::class_<ConstraintOptions::Eval>(m, "ConstraintOptionsEval")
        .def_readwrite("ddx", &ConstraintOptions::Eval::ddx, "True if evaluating the Jacobian matrix of c(x, p) with respect to x is needed")
        .def_readwrite("ddp", &ConstraintOptions::Eval::ddp, "True if evaluating the Jacobian matrix of c(x, p) with respect to p is needed")
        ;

    py::class_<ConstraintOptions>(m, "ConstraintOptions")
        .def_readonly("eval", &ConstraintOptions::eval, "The objective function components that need to be evaluated.")
        .def_readonly("ibasicvars", &ConstraintOptions::ibasicvars, "The indices of the basic variables in x.")
        ;

    py::class_<ConstraintFunction>(m, "ConstraintFunction")
        .def(py::init<const ConstraintFunction::Signature4py&>())
        .def("__call__", &ConstraintFunction::operator())
        ;

    py::implicitly_convertible<ConstraintFunction::Signature4py, ConstraintFunction>();
}

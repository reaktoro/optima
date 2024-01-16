// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include "pybind11.hxx"

// Optima includes
#include <Optima/ConstraintFunction.hpp>
#include <Optima/Utils.hpp>
using namespace Optima;

void exportConstraintResult(py::module& m)
{
    auto get_val = [](ConstraintResult& s) -> VectorRef { return s.val; };
    auto get_ddx = [](ConstraintResult& s) -> MatrixRef { return s.ddx; };
    auto get_ddp = [](ConstraintResult& s) -> MatrixRef { return s.ddp; };
    auto get_ddc = [](ConstraintResult& s) -> MatrixRef { return s.ddc; };

    auto set_val = [](ConstraintResult& s, VectorView val)    { assignOrError(s.val, val); };
    auto set_ddx = [](ConstraintResult& s, MatrixView4py ddx) { assignOrError(s.ddx, ddx); };
    auto set_ddp = [](ConstraintResult& s, MatrixView4py ddp) { assignOrError(s.ddp, ddp); };
    auto set_ddc = [](ConstraintResult& s, MatrixView4py ddc) { assignOrError(s.ddc, ddc); };

    py::class_<ConstraintResult>(m, "ConstraintResult")
        .def_property("val", get_val, set_val)
        .def_property("ddx", get_ddx, set_ddx)
        .def_property("ddp", get_ddp, set_ddp)
        .def_property("ddc", get_ddc, set_ddc)
        .def_readwrite("ddx4basicvars", &ConstraintResult::ddx4basicvars)
        .def_readwrite("succeeded", &ConstraintResult::succeeded)
        .def("resize", &ConstraintResult::resize)
        ;
}

void exportConstraintResultRef(py::module& m)
{
    auto get_val           = [](ConstraintResultRef& s) -> VectorRef { return s.val; };
    auto get_ddx           = [](ConstraintResultRef& s) -> MatrixRef { return s.ddx; };
    auto get_ddp           = [](ConstraintResultRef& s) -> MatrixRef { return s.ddp; };
    auto get_ddc           = [](ConstraintResultRef& s) -> MatrixRef { return s.ddc; };
    auto get_ddx4basicvars = [](ConstraintResultRef& s) -> bool& { return s.ddx4basicvars; };
    auto get_succeeded     = [](ConstraintResultRef& s) -> bool& { return s.succeeded; };

    auto set_val           = [](ConstraintResultRef& s, VectorView val)    { assignOrError(s.val, val); };
    auto set_ddx           = [](ConstraintResultRef& s, MatrixView4py ddx) { assignOrError(s.ddx, ddx); };
    auto set_ddp           = [](ConstraintResultRef& s, MatrixView4py ddp) { assignOrError(s.ddp, ddp); };
    auto set_ddc           = [](ConstraintResultRef& s, MatrixView4py ddc) { assignOrError(s.ddc, ddc); };
    auto set_ddx4basicvars = [](ConstraintResultRef& s, bool ddx4basicvars) { s.ddx4basicvars = ddx4basicvars; };
    auto set_succeeded     = [](ConstraintResultRef& s, bool succeeded) { s.succeeded = succeeded; };

    py::class_<ConstraintResultRef>(m, "ConstraintResultRef")
        .def_property("val", get_val, set_val)
        .def_property("ddx", get_ddx, set_ddx)
        .def_property("ddp", get_ddp, set_ddp)
        .def_property("ddc", get_ddc, set_ddc)
        .def_property("ddx4basicvars", get_ddx4basicvars, set_ddx4basicvars)
        .def_property("succeeded", get_succeeded, set_succeeded)
        ;

    py::implicitly_convertible<ConstraintResult, ConstraintResultRef>();
}

void exportConstraintFunction(py::module& m)
{
    exportConstraintResult(m);
    exportConstraintResultRef(m);

    py::class_<ConstraintOptions::Eval>(m, "ConstraintOptionsEval")
        .def_readwrite("ddx", &ConstraintOptions::Eval::ddx, "True if evaluating the Jacobian matrix of q(x, p, c) with respect to x is needed")
        .def_readwrite("ddp", &ConstraintOptions::Eval::ddp, "True if evaluating the Jacobian matrix of q(x, p, c) with respect to p is needed")
        .def_readwrite("ddc", &ConstraintOptions::Eval::ddc, "True if evaluating the Jacobian matrix of q(x, p, c) with respect to c is needed")
        ;

    py::class_<ConstraintOptions>(m, "ConstraintOptions")
        .def_readonly("eval", &ConstraintOptions::eval, "The objective function components that need to be evaluated.")
        .def_readonly("ibasicvars", &ConstraintOptions::ibasicvars, "The indices of the basic variables in x.")
        ;

    py::class_<ConstraintFunction>(m, "ConstraintFunction")
        .def(py::init<const ConstraintFunction::Signature4py&>())
        .def("__call__", &ConstraintFunction::operator())
        .def("initialized", &ConstraintFunction::initialized)
        ;

    py::implicitly_convertible<ConstraintFunction::Signature4py, ConstraintFunction>();
}

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
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/Utils.hpp>
using namespace Optima;

void exportObjectiveResult(py::module& m)
{
    auto get_fx  = [](ObjectiveResult& s) -> VectorRef { return s.fx; };
    auto get_fxx = [](ObjectiveResult& s) -> MatrixRef { return s.fxx; };
    auto get_fxp = [](ObjectiveResult& s) -> MatrixRef { return s.fxp; };
    auto get_fxc = [](ObjectiveResult& s) -> MatrixRef { return s.fxc; };

    auto set_fx  = [](ObjectiveResult& s, VectorView fx)     { assignOrError(s.fx, fx); };
    auto set_fxx = [](ObjectiveResult& s, MatrixView4py fxx) { assignOrError(s.fxx, fxx); };
    auto set_fxp = [](ObjectiveResult& s, MatrixView4py fxp) { assignOrError(s.fxp, fxp); };
    auto set_fxc = [](ObjectiveResult& s, MatrixView4py fxc) { assignOrError(s.fxc, fxc); };

    py::class_<ObjectiveResult>(m, "ObjectiveResult")
        .def_readwrite("f", &ObjectiveResult::f)
        .def_property("fx", get_fx, set_fx)
        .def_property("fxx", get_fxx, set_fxx)
        .def_property("fxp", get_fxp, set_fxp)
        .def_property("fxc", get_fxc, set_fxc)
        .def_readwrite("diagfxx", &ObjectiveResult::diagfxx)
        .def_readwrite("fxx4basicvars", &ObjectiveResult::fxx4basicvars)
        .def_readwrite("succeeded", &ObjectiveResult::succeeded)
        .def("resize", &ObjectiveResult::resize)
        ;
}

void exportObjectiveResultRef(py::module& m)
{
    auto get_f             = [](ObjectiveResultRef& s) -> double& { return s.f; };
    auto get_fx            = [](ObjectiveResultRef& s) -> VectorRef { return s.fx; };
    auto get_fxx           = [](ObjectiveResultRef& s) -> MatrixRef { return s.fxx; };
    auto get_fxp           = [](ObjectiveResultRef& s) -> MatrixRef { return s.fxp; };
    auto get_fxc           = [](ObjectiveResultRef& s) -> MatrixRef { return s.fxc; };
    auto get_diagfxx       = [](ObjectiveResultRef& s) -> bool& { return s.diagfxx; };
    auto get_fxx4basicvars = [](ObjectiveResultRef& s) -> bool& { return s.fxx4basicvars; };
    auto get_succeeded     = [](ObjectiveResultRef& s) -> bool& { return s.succeeded; };

    auto set_f             = [](ObjectiveResultRef& s, double f) { s.f = f; };
    auto set_fx            = [](ObjectiveResultRef& s, VectorView fx)     { assignOrError(s.fx, fx); };
    auto set_fxx           = [](ObjectiveResultRef& s, MatrixView4py fxx) { assignOrError(s.fxx, fxx); };
    auto set_fxp           = [](ObjectiveResultRef& s, MatrixView4py fxp) { assignOrError(s.fxp, fxp); };
    auto set_fxc           = [](ObjectiveResultRef& s, MatrixView4py fxc) { assignOrError(s.fxc, fxc); };
    auto set_diagfxx       = [](ObjectiveResultRef& s, bool diagfxx) { s.diagfxx = diagfxx; };
    auto set_fxx4basicvars = [](ObjectiveResultRef& s, bool fxx4basicvars) { s.fxx4basicvars = fxx4basicvars; };
    auto set_succeeded     = [](ObjectiveResultRef& s, bool succeeded) { s.succeeded = succeeded; };

    py::class_<ObjectiveResultRef>(m, "ObjectiveResultRef")
        .def_property("f"            , get_f            , set_f)
        .def_property("fx"           , get_fx           , set_fx)
        .def_property("fxx"          , get_fxx          , set_fxx)
        .def_property("fxp"          , get_fxp          , set_fxp)
        .def_property("fxc"          , get_fxc          , set_fxc)
        .def_property("diagfxx"      , get_diagfxx      , set_diagfxx)
        .def_property("fxx4basicvars", get_fxx4basicvars, set_fxx4basicvars)
        .def_property("succeeded"    , get_succeeded    , set_succeeded)
        ;

    py::implicitly_convertible<ObjectiveResult, ObjectiveResultRef>();
}

void exportObjectiveFunction(py::module& m)
{
    exportObjectiveResult(m);
    exportObjectiveResultRef(m);

    py::class_<ObjectiveOptions::Eval>(m, "ObjectiveOptionsEval")
        .def_readwrite("fxx", &ObjectiveOptions::Eval::fxx, "True if evaluating the Jacobian matrix fxx is needed.")
        .def_readwrite("fxp", &ObjectiveOptions::Eval::fxp, "True if evaluating the Jacobian matrix fxp is needed.")
        .def_readwrite("fxc", &ObjectiveOptions::Eval::fxc, "True if evaluating the Jacobian matrix fxc is needed.")
        ;

    py::class_<ObjectiveOptions>(m, "ObjectiveOptions")
        .def_readonly("eval", &ObjectiveOptions::eval, "The objective function components that need to be evaluated.")
        .def_readonly("ibasicvars", &ObjectiveOptions::ibasicvars, "The indices of the basic variables in x.")
        ;

    py::class_<ObjectiveFunction>(m, "ObjectiveFunction")
        .def(py::init<const ObjectiveFunction::Signature4py&>())
        .def("__call__", &ObjectiveFunction::operator())
        .def("initialized", &ObjectiveFunction::initialized)
        ;

    py::implicitly_convertible<ObjectiveFunction::Signature4py, ObjectiveFunction>();
}

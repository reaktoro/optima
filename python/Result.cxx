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
#include <pybind11/operators.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Result.hpp>
using namespace Optima;

void exportResult(py::module& m)
{
    const auto success1 = static_cast<void(Result::*)(bool)>(&Result::success);
    const auto success2 = static_cast<bool(Result::*)() const>(&Result::success);

    py::class_<Result>(m, "Result")
        .def(py::init<>())
        .def("success", success1)
        .def("success", success2)
        .def("time", &Result::time)
        .def("start", &Result::start, py::return_value_policy::reference)
        .def("stop", &Result::stop, py::return_value_policy::reference)
        .def(py::self += py::self)
        .def(py::self + py::self)
        ;
}

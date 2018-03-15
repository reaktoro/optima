// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <Optima/OptimumParams.hpp>
#include <Optima/OptimumStructure.hpp>
using namespace Optima;

void exportOptimumParams(py::module& m)
{
    auto get_b = static_cast<VectorConstRef (OptimumParams::*)() const>(&OptimumParams::b);
    auto get_lowerBounds = static_cast<VectorConstRef (OptimumParams::*)() const>(&OptimumParams::lowerBounds);
    auto get_upperBounds = static_cast<VectorConstRef (OptimumParams::*)() const>(&OptimumParams::upperBounds);
    auto get_fixedValues = static_cast<VectorConstRef (OptimumParams::*)() const>(&OptimumParams::fixedValues);

    auto set_b = static_cast<VectorRef (OptimumParams::*)()>(&OptimumParams::b);
    auto set_lowerBounds = static_cast<VectorRef (OptimumParams::*)()>(&OptimumParams::b);
    auto set_upperBounds = static_cast<VectorRef (OptimumParams::*)()>(&OptimumParams::b);
    auto set_fixedValues = static_cast<VectorRef (OptimumParams::*)()>(&OptimumParams::b);

    py::class_<OptimumParams>(m, "OptimumParams")
        .def(py::init<const OptimumStructure&>())
        .def_property("b", get_b, set_b)
        .def_property("lowerBounds", get_lowerBounds, set_lowerBounds)
        .def_property("upperBounds", get_upperBounds, set_upperBounds)
        .def_property("fixedValues", get_fixedValues, set_fixedValues)
        ;
}

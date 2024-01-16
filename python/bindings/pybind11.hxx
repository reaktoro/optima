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

#pragma once

//----------------------------------------------------------------------------------------------------------------
// Add defines above to allow pybind11 packages produced with different compilers/versions to be used together.
// https://github.com/pybind/pybind11/pull/2602
//----------------------------------------------------------------------------------------------------------------
#define PYBIND11_COMPILER_TYPE ""
#define PYBIND11_STDLIB ""
#define PYBIND11_BUILD_ABI ""
//----------------------------------------------------------------------------------------------------------------

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
namespace py = pybind11;

/// Used to indicate that the returned object of a method and the parent/this
/// object must both be kept alive while the other is alive.
#define PYBIND_ENSURE_MUTUAL_EXISTENCE \
    py::keep_alive<1, 0>(), py::keep_alive<0, 1>()

/// Used to indicate that the k-th argument in a method should be kept alive in Python.
/// Note: pybind11's numbering convention for `py::keep_alive` starts with `2`
/// for arguments. Index `1` denotes the `this` pointer and `0` the returned object.
/// Here, however, `0` denotes the first argument in C++, which is not the `this` pointer.
template<size_t k>
using keep_argument_alive = py::keep_alive<1, k + 2>;

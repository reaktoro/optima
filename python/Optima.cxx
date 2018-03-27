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
namespace py = pybind11;

void exportEigen(py::module& m);
void exportCanonicalizer(py::module& m);
void exportOutputter(py::module& m);
void exportResult(py::module& m);
void exportObjective(py::module& m);
void exportOptimumOptions(py::module& m);
void exportOptimumParams(py::module& m);
void exportOptimumProblem(py::module& m);
void exportOptimumResult(py::module& m);
void exportOptimumSolver(py::module& m);
void exportOptimumState(py::module& m);
void exportOptimumStepper(py::module& m);
void exportOptimumStructure(py::module& m);
void exportSaddlePointMatrix(py::module& m);
void exportSaddlePointOptions(py::module& m);
void exportSaddlePointSolver(py::module& m);
void exportIpSaddlePointSolver(py::module& m);
void exportIpSaddlePointMatrix(py::module& m);
void exportTiming(py::module& m);
void exportUtils(py::module& m);
void exportVariantMatrix(py::module& m);

PYBIND11_MODULE(optima, m)
{
    exportEigen(m);
    exportCanonicalizer(m);
    exportOutputter(m);
    exportResult(m);
    exportObjective(m);
    exportOptimumOptions(m);
    exportOptimumParams(m);
    exportOptimumProblem(m);
    exportOptimumResult(m);
    exportOptimumSolver(m);
    exportOptimumState(m);
    exportOptimumStepper(m);
    exportOptimumStructure(m);
    exportSaddlePointMatrix(m);
    exportSaddlePointOptions(m);
    exportSaddlePointSolver(m);
    exportIpSaddlePointSolver(m);
    exportIpSaddlePointMatrix(m);
    exportTiming(m);
    exportUtils(m);
    exportVariantMatrix(m);
}

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
namespace py = pybind11;

void exportEigen(py::module& m);
void exportCanonicalizer(py::module& m);
void exportIndexUtils(py::module& m);
void exportOutputter(py::module& m);
void exportPartition(py::module& m);
void exportObjective(py::module& m);
void exportOptions(py::module& m);
void exportParams(py::module& m);
void exportProblem(py::module& m);
void exportResult(py::module& m);
void exportSolver(py::module& m);
void exportState(py::module& m);
void exportStepper(py::module& m);
void exportStructure(py::module& m);
void exportSaddlePointMatrix(py::module& m);
void exportSaddlePointOptions(py::module& m);
void exportSaddlePointSolver(py::module& m);
void exportIpSaddlePointSolver(py::module& m);
void exportIpSaddlePointMatrix(py::module& m);
void exportTiming(py::module& m);
void exportUtils(py::module& m);

PYBIND11_MODULE(optima, m)
{
    exportEigen(m);
    exportCanonicalizer(m);
    exportIndexUtils(m);
    exportOutputter(m);
    exportPartition(m);
    exportObjective(m);
    exportOptions(m);
    exportParams(m);
    exportProblem(m);
    exportResult(m);
    exportState(m);
    exportStepper(m);
    exportStructure(m);
    exportSolver(m);
    exportSaddlePointMatrix(m);
    exportSaddlePointOptions(m);
    exportSaddlePointSolver(m);
    exportIpSaddlePointSolver(m);
    exportIpSaddlePointMatrix(m);
    exportTiming(m);
    exportUtils(m);
}

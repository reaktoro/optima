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

void exportEigen(py::module& m);
void exportBacktrackSearchOptions(py::module& m);
void exportConstants(py::module& m);
void exportConvergenceOptions(py::module& m);
void exportCanonicalDims(py::module& m);
void exportCanonicalizer(py::module& m);
void exportCanonicalMatrix(py::module& m);
void exportCanonicalVector(py::module& m);
void exportErrorStatusOptions(py::module& m);
void exportConstraintFunction(py::module& m);
void exportDims(py::module& m);
void exportEchelonizer(py::module& m);
void exportEchelonizerExtended(py::module& m);
void exportEchelonizerW(py::module& m);
void exportIndex(py::module& m);
void exportIndexUtils(py::module& m);
void exportLineSearchOptions(py::module& m);
void exportLinearSolver(py::module& m);
void exportLinearSolverOptions(py::module& m);
void exportLU(py::module& m);
void exportMasterDims(py::module& m);
void exportMasterProblem(py::module& m);
void exportMasterSensitivity(py::module& m);
void exportMasterSolver(py::module& m);
void exportMasterState(py::module& m);
void exportMasterMatrix(py::module& m);
void exportMasterMatrixOps(py::module& m);
void exportMasterVector(py::module& m);
void exportMatrixViewH(py::module& m);
void exportMatrixViewRWQ(py::module& m);
void exportMatrixViewV(py::module& m);
void exportMatrixViewW(py::module& m);
void exportNewtonStep(py::module& m);
void exportNewtonStepOptions(py::module& m);
void exportObjectiveFunction(py::module& m);
void exportOutputter(py::module& m);
void exportOptions(py::module& m);
void exportProblem(py::module& m);
void exportResidualFunction(py::module& m);
void exportResidualVector(py::module& m);
void exportResourcesFunction(py::module& m);
void exportResult(py::module& m);
void exportSensitivity(py::module& m);
void exportSensitivitySolver(py::module& m);
void exportSolver(py::module& m);
void exportStablePartition(py::module& m);
void exportStability(py::module& m);
void exportState(py::module& m);
void exportTiming(py::module& m);
void exportUtils(py::module& m);

PYBIND11_MODULE(optima4py, m)
{
    exportEigen(m);
    exportBacktrackSearchOptions(m);
    exportConstants(m);
    exportConvergenceOptions(m);
    exportCanonicalDims(m);
    exportCanonicalizer(m);
    exportCanonicalMatrix(m);
    exportCanonicalVector(m);
    exportErrorStatusOptions(m);
    exportConstraintFunction(m);
    exportDims(m);
    exportEchelonizer(m);
    exportEchelonizerExtended(m);
    exportEchelonizerW(m);
    exportIndex(m);
    exportIndexUtils(m);
    exportLineSearchOptions(m);
    exportLinearSolver(m);
    exportLinearSolverOptions(m);
    exportLU(m);
    exportMasterDims(m);
    exportMasterProblem(m);
    exportMasterSensitivity(m);
    exportMasterSolver(m);
    exportMasterState(m);
    exportMasterMatrix(m);
    exportMasterMatrixOps(m);
    exportMasterVector(m);
    exportMatrixViewH(m);
    exportMatrixViewRWQ(m);
    exportMatrixViewV(m);
    exportMatrixViewW(m);
    exportNewtonStep(m);
    exportNewtonStepOptions(m);
    exportObjectiveFunction(m);
    exportOutputter(m);
    exportOptions(m);
    exportProblem(m);
    exportResidualFunction(m);
    exportResidualVector(m);
    exportResourcesFunction(m);
    exportResult(m);
    exportSensitivity(m);
    exportSensitivitySolver(m);
    exportSolver(m);
    exportStablePartition(m);
    exportStability(m);
    exportState(m);
    exportTiming(m);
    exportUtils(m);
}

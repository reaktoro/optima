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
#include <Optima/Utils.hpp>
using namespace Optima;

void exportUtils(py::module& m)
{
    py::enum_<MatrixStructure>(m, "MatrixStructure")
        .value("Dense", MatrixStructure::Dense)
        .value("Diagonal", MatrixStructure::Diagonal)
        .value("Zero", MatrixStructure::Zero)
        ;

    auto largestStep1 = static_cast<double(*)(const Vector&, const Vector&)>(&largestStep);
    auto largestStep2 = static_cast<double(*)(const Vector&, const Vector&, const Vector&, const Vector&)>(&largestStep);

    auto multiplyMatrixVectorWithoutResidualRoundOffError = [](MatrixView4py A, VectorView x)
    {
        return Optima::multiplyMatrixVectorWithoutResidualRoundOffError(A, x);
    };

    m.def("largestStep", largestStep1);
    m.def("largestStep", largestStep2);
    m.def("performConservativeStep", &performConservativeStep);
    m.def("performAggressiveStep", &performAggressiveStep);
    m.def("lessThan", &lessThan);
    m.def("greaterThan", &greaterThan);
    m.def("infinity", &infinity);
    m.def("bfgs", &bfgs);
    m.def("minimizeGoldenSectionSearch", &minimizeGoldenSectionSearch);
    m.def("minimizeBrent", &minimizeBrent);
    m.def("inverseShermanMorrison", &inverseShermanMorrison);
    m.def("rational", &rational);
    m.def("rationalize", &rationalize);
    m.def("multiplyMatrixVectorWithoutResidualRoundOffError", multiplyMatrixVectorWithoutResidualRoundOffError);
    m.def("matrixStructure", &matrixStructure);
    m.def("isZeroMatrix", &isZeroMatrix);
    m.def("isDiagonalMatrix", &isDiagonalMatrix);
    m.def("isDenseMatrix", &isDenseMatrix);
    m.def("isDenseMatrix", &isDenseMatrix);
}

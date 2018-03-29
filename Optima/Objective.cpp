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

#include "Objective.hpp"

// Optima includes
#include <Optima/OptimumStructure.hpp>
using namespace Eigen;

namespace Optima {

ObjectiveResult::ObjectiveResult()
: f(0.0), failed(false)
{}

ObjectiveResult::ObjectiveResult(const OptimumStructure& structure)
: f(0.0), g(zeros(structure.numVariables())), failed(false)
{
    switch(structure.structureHessianMatrix())
    {
    case MatrixStructure::Dense:
        H.setDense(structure.numVariables());
        H.dense.fill(0.0);
        break;
    case MatrixStructure::Diagonal:
    case MatrixStructure::Zero:
        H.setDiagonal(structure.numVariables());
        H.diagonal.fill(0.0);
        break;
    }
}

} // namespace Optima

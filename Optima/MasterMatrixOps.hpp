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

// Optima includes
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterVector.hpp>

namespace Optima {

/// Used to represent the transpose expression of a master matrix.
struct MasterMatrixTrExpr
{
    const MasterMatrix& M; ///< The underlying master matrix in the transpose expression.
};

/// Return a transpose representation of a master matrix.
inline auto tr(const MasterMatrix& M) -> MasterMatrixTrExpr { return { M }; }

/// Return the product of a master matrix and a master vector.
auto operator*(const MasterMatrix& M, const MasterVectorView& u) -> MasterVector;

/// Return the product of a master matrix transpose and a master vector.
auto operator*(const MasterMatrixTrExpr& trM, const MasterVectorView& u) -> MasterVector;

} // namespace Optima

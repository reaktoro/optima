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

#pragma once

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent block matrix \eq{V} in @ref JacobianMatrix.
class JacobianBlockV
{
private:
    Matrix _Vpx; ///< The Vpx block matrix in V = [Vpx Vpp].
    Matrix _Vpp; ///< The Vpp block matrix in V = [Vpx Vpp].

public:
    /// Construct a JacobianBlockV instance.
    JacobianBlockV(Index nx, Index np);

    /// Construct a JacobianBlockV instance.
    JacobianBlockV(MatrixConstRef Vpx, MatrixConstRef Vpp);

    /// Construct a copy of a JacobianBlockV instance.
    JacobianBlockV(const JacobianBlockV& other);

    /// Assign a JacobianBlockV instance to this.
    auto operator=(JacobianBlockV other) -> JacobianBlockV& = delete;

    /// The reference to the Vpx block matrix in V = [Vpx Vpp].
    MatrixRef Vpx;

    /// The reference to the Vpp block matrix in V = [Vpx Vpp].
    MatrixRef Vpp;
};

} // namespace Optima

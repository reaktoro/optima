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

/// Used to represent block matrix \eq{H} in @ref JacobianMatrix.
class JacobianBlockH
{
private:
    Matrix _Hxx; ///< The Hxx block matrix in H = [Hxx Hxp].
    Matrix _Hxp; ///< The Hxp block matrix in H = [Hxx Hxp].
    bool isdiag; ///< The flag that indicates wether Hxx is diagonal.

public:
    /// Construct a JacobianBlockH instance.
    JacobianBlockH(Index nx, Index np);

    /// Construct a JacobianBlockH instance.
    JacobianBlockH(MatrixConstRef Hxx, MatrixConstRef Hxp);

    /// Construct a copy of a JacobianBlockH instance.
    JacobianBlockH(const JacobianBlockH& other);

    /// Assign a JacobianBlockH instance to this.
    auto operator=(JacobianBlockH other) -> JacobianBlockH& = delete;

    /// Return true if Hxx has non-zero values only along its diagonal.
    auto isHxxDiagonal() const -> bool;

    /// Specify whether Hxx has diagonal structure.
    auto isHxxDiagonal(bool enable) -> bool;

    MatrixRef Hxx; ///< The reference to the Hxx block matrix in H = [Hxx Hxp].
    MatrixRef Hxp; ///< The reference to the Hxp block matrix in H = [Hxx Hxp].
};

} // namespace Optima

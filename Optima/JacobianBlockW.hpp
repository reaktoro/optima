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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent block matrix \eq{W} in @ref JacobianMatrix.
class JacobianBlockW
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a JacobianBlockW instance.
    /// @param nx The number of columns in Ax and Jx
    /// @param np The number of columns in Ap and Jp
    /// @param ny The number of rows in Ax and Ap
    /// @param nz The number of rows in Jx and Jp
    /// @param Ax The matrix Ax in W = [Ax Ap; Jx Jp]
    /// @param Ap The matrix Ap in W = [Ax Ap; Jx Jp]
    JacobianBlockW(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap);

    /// Construct a copy of a JacobianBlockW instance.
    JacobianBlockW(const JacobianBlockW& other);

    /// Destroy this JacobianBlockW instance.
    virtual ~JacobianBlockW();

    /// Assign a JacobianBlockW instance to this.
    auto operator=(JacobianBlockW other) -> JacobianBlockW& = delete;

    /// Update the matrix block and its canonical form with new Jx and Jp matrices.
    auto update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void;

    /// The matrix components in the canonical form of matrix W.
    struct CanonicalForm
    {
        MatrixConstRef  R;   ///< The canonicalizer matrix of W so that RWQ = [Ibb Sbn Sbp] with Q = (jb, jn).
        MatrixConstRef  Sbn; ///< The matrix Sbn in the canonical form of W.
        MatrixConstRef  Sbp; ///< The matrix Sbp in the canonical form of W.
        IndicesConstRef jb;  ///< The indices of the basic variables in the canonical form of W.
        IndicesConstRef jn;  ///< The indices of the non-basic variables in the canonical form of W.
    };

    /// Return a view to the components of the canonical form of matrix W.
    auto canonicalForm() const -> CanonicalForm;

    MatrixConstRef Ax; ///< The reference to the Ax block matrix in W = [Ax Ap; Jx Jp].
    MatrixConstRef Ap; ///< The reference to the Ap block matrix in W = [Ax Ap; Jx Jp].
    MatrixConstRef Jx; ///< The reference to the Jx block matrix in W = [Ax Ap; Jx Jp].
    MatrixConstRef Jp; ///< The reference to the Jp block matrix in W = [Ax Ap; Jx Jp].
    MatrixConstRef Wx; ///< The reference to the Wp block matrix in W = [Wx Wp] = [Ax Ap; Jx Jp].
    MatrixConstRef Wp; ///< The reference to the Wp block matrix in W = [Wx Wp] = [Ax Ap; Jx Jp].
};

} // namespace Optima

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

/// Used to represent the echelon form *RWQ = [Ibb Sbn Sbp]* of matrix *W = [Wx Wp] = [Ax Ap; Jx Jp]*.
struct MatrixViewRWQ
{
    MatrixConstRef R;   ///< The echelonizer matrix of W so that *RWQ = [Ibb Sbn Sbp]* with *Q = (jb, jn)*.
    MatrixConstRef Sbn; ///< The matrix *Sbn* in the echelon form of *W*.
    MatrixConstRef Sbp; ///< The matrix *Sbp* in the echelon form of *W*.
    IndicesConstRef jb; ///< The indices of the basic variables in the echelon form of *W*.
    IndicesConstRef jn; ///< The indices of the non-basic variables in the echelon form of *W*.
};

/// Used to compute the echelon form of matrix of matrix *W = [Ax Ap; Jx Jp]*.
class MatrixRWQ
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a MatrixRWQ instance.
    /// @param nx The number of columns in Ax and Jx
    /// @param np The number of columns in Ap and Jp
    /// @param ny The number of rows in Ax and Ap
    /// @param nz The number of rows in Jx and Jp
    /// @param Ax The matrix Ax in W = [Ax Ap; Jx Jp]
    /// @param Ap The matrix Ap in W = [Ax Ap; Jx Jp]
    MatrixRWQ(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap);

    /// Construct a copy of a MatrixRWQ instance.
    MatrixRWQ(const MatrixRWQ& other);

    /// Destroy this MatrixRWQ instance.
    virtual ~MatrixRWQ();

    /// Assign a MatrixRWQ instance to this.
    auto operator=(MatrixRWQ other) -> MatrixRWQ& = delete;

    /// Update the echelon form of matrix *W*.
    auto update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void;

    /// Return an immutable view to the echelon form of *W*.
    auto view() const -> MatrixViewRWQ;
};

} // namespace Optima

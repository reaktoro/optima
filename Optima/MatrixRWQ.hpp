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
#include <Optima/MasterDims.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent a view to matrix *W = [Wx Wp] = [Ax Ap; Jx Jp]*.
struct MatrixViewW
{
    MatrixConstRef Wx;  ///< The view to sub-matrix *Wx* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixConstRef Wp;  ///< The view to sub-matrix *Wp* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixConstRef Ax;  ///< The view to sub-matrix *Ax* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixConstRef Ap;  ///< The view to sub-matrix *Ap* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixConstRef Jx;  ///< The view to sub-matrix *Jx* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixConstRef Jp;  ///< The view to sub-matrix *Jp* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
};

/// Used to represent the echelon form *RWQ = [Ibb Sbn Sbp]* of matrix *W*.
struct MatrixViewRWQ
{
    MatrixConstRef R;   ///< The echelonizer matrix of W so that *RWQ = [Ibb Sbn Sbp]* with *Q = (jb, jn)*.
    MatrixConstRef Sbn; ///< The matrix *Sbn* in the echelon form of *W*.
    MatrixConstRef Sbp; ///< The matrix *Sbp* in the echelon form of *W*.
    IndicesConstRef jb; ///< The indices of the basic variables in the echelon form of *W*.
    IndicesConstRef jn; ///< The indices of the non-basic variables in the echelon form of *W*.
};

/// Used to compute the echelon form of matrix *W = [Ax Ap; Jx Jp]*.
class MatrixRWQ
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a MatrixRWQ instance.
    /// @param dims The dimensions of the master variables
    /// @param Ax The matrix *Ax* in *W = [Ax Ap; Jx Jp]*.
    /// @param Ap The matrix *Ap* in *W = [Ax Ap; Jx Jp]*.
    MatrixRWQ(const MasterDims& dims, MatrixConstRef Ax, MatrixConstRef Ap);

    /// Construct a copy of a MatrixRWQ instance.
    MatrixRWQ(const MatrixRWQ& other);

    /// Destroy this MatrixRWQ instance.
    virtual ~MatrixRWQ();

    /// Assign a MatrixRWQ instance to this.
    auto operator=(MatrixRWQ other) -> MatrixRWQ& = delete;

    /// Update the echelon form of matrix *W*.
    auto update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void;

    /// Return the dimensions of the master variables.
    auto dims() const -> MasterDims;

    /// Return an immutable view to matrix *W*.
    auto asMatrixViewW() const -> MatrixViewW;

    /// Return an immutable view to the echelon form of *W*.
    auto asMatrixViewRWQ() const -> MatrixViewRWQ;

    /// Convert this MatrixRWQ object into a MatrixViewW object.
    explicit operator MatrixViewW() const;

    /// Convert this MatrixRWQ object into a MatrixViewRWQ object.
    explicit operator MatrixViewRWQ() const;
};

} // namespace Optima

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
#include <Optima/MasterMatrix.hpp>

namespace Optima {

/// The dimension details of a master matrix in its canonical form.
struct CanonicalDims
{
    Index nx;  ///< The number of variables x.
    Index np;  ///< The number of variables p.
    Index ny;  ///< The number of variables y.
    Index nz;  ///< The number of variables z.
    Index nw;  ///< The number of variables w = (y, z).
    Index ns;  ///< The number of stable variables in x.
    Index nu;  ///< The number of unstable variables in x.
    Index nb;  ///< The number of basic variables in x.
    Index nn;  ///< The number of non-basic variables in x.
    Index nl;  ///< The number of linearly dependent rows in Wx = [Ax; Jx].
    Index nbs; ///< The number of stable basic variables.
    Index nbu; ///< The number of unstable basic variables.
    Index nns; ///< The number of stable non-basic variables.
    Index nnu; ///< The number of unstable non-basic variables.
    Index nbe; ///< The number of stable explicit basic variables.
    Index nbi; ///< The number of stable implicit basic variables.
    Index nne; ///< The number of stable explicit non-basic variables.
    Index nni; ///< The number of stable implicit non-basic variables.
};

/// Used to represent the canonical form of a master matrix.
struct CanonicalMatrixView
{
    CanonicalDims dims;   ///< The dimension details of a master matrix in its canonical form.
    MatrixConstRef Hss;   ///< The matrix Hss in the canonical master matrix.
    MatrixConstRef Hsp;   ///< The matrix Hsp in the canonical master matrix.
    MatrixConstRef Vps;   ///< The matrix Vps in the canonical master matrix.
    MatrixConstRef Vpp;   ///< The matrix Vpp in the canonical master matrix.
    MatrixConstRef Sbsns; ///< The matrix Sbsns in the canonical master matrix.
    MatrixConstRef Sbsp;  ///< The matrix Sbsp in the canonical master matrix.
    MatrixConstRef Rbs;   ///< The matrix Rbs in the echelonizer matrix R = [Rbs; 0].
    IndicesConstRef jb;   ///< The indices of the basic variables ordered as jb = (jbs).
    IndicesConstRef jn;   ///< The indices of the non-basic variables ordered as jn = (jns, jnu).
    IndicesConstRef js;   ///< The indices of the stable variables ordered as js = (jbs, jns).
    IndicesConstRef ju;   ///< The indices of the unstable variables ordered as ju = (jbu, jnu).
};

/// Used to assemble the canonical form of a master matrix.
class CanonicalMatrix
{
public:
    /// Construct a CanonicalMatrix instance.
    CanonicalMatrix(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a CanonicalMatrix instance.
    CanonicalMatrix(const CanonicalMatrix& other);

    /// Destroy this CanonicalMatrix instance.
    virtual ~CanonicalMatrix();

    /// Assign a CanonicalMatrix instance to this.
    auto operator=(CanonicalMatrix other) -> CanonicalMatrix&;

    /// Assemble the canonical form of the master matrix.
    auto update(const MasterMatrix& M) -> void;

    /// Return an immutable view to the canonical form of a master matrix.
    auto view() const -> CanonicalMatrixView;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

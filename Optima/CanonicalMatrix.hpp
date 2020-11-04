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
#include <Optima/CanonicalDims.hpp>
#include <Optima/MasterDims.hpp>
#include <Optima/MasterMatrix.hpp>

namespace Optima {

/// Used to represent the canonical form of a master matrix.
struct CanonicalMatrixView
{
    CanonicalDims dims;   ///< The dimension details of the canonical master matrix.
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
    CanonicalMatrix(const MasterDims& dims);

    /// Construct a CanonicalMatrix instance.
    CanonicalMatrix(const MasterMatrix& M);

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

    /// Convert this CanonicalMatrix object into a CanonicalMatrixView object.
    operator CanonicalMatrixView() const;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

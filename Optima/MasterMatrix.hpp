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
#include <Optima/MasterDims.hpp>
#include <Optima/MatrixViewH.hpp>
#include <Optima/MatrixViewV.hpp>
#include <Optima/MatrixViewW.hpp>
#include <Optima/MatrixViewRWQ.hpp>

namespace Optima {

/// Used to represent a master matrix.
struct MasterMatrix
{
    /// The dimension details of the master matrix.
    const MasterDims dims;

    /// The matrix *H = [Hxx Hxp]* in the master matrix.
    const MatrixViewH H;

    /// The matrix *V = [Vpx Vpp]* in the master matrix.
    const MatrixViewV V;

    /// The matrix *W = [Ax Ap; Jx Jp]* in the master matrix.
    const MatrixViewW W;

    /// The echelon form *RWQ = [Ibb Sbn Sbp]* of matrix *W* in the master matrix.
    const MatrixViewRWQ RWQ;

    /// The indices of the stable variables in *x*.
    const IndicesView js;

    /// The indices of the unstable variables in *x*.
    const IndicesView ju;

    /// Convert this MasterMatrix object into a Matrix object.
    operator Matrix() const;
};

} // namespace Optima

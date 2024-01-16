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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/CanonicalDims.hpp>
#include <Optima/MasterDims.hpp>
#include <Optima/MasterMatrix.hpp>

namespace Optima {

/// Used to represent the canonical form of a master matrix.
struct CanonicalMatrix
{
    CanonicalDims dims; ///< The dimension details of the canonical master matrix.
    MatrixView Hss;     ///< The matrix Hss in the canonical master matrix.
    MatrixView Hsp;     ///< The matrix Hsp in the canonical master matrix.
    MatrixView Vps;     ///< The matrix Vps in the canonical master matrix.
    MatrixView Vpp;     ///< The matrix Vpp in the canonical master matrix.
    MatrixView Sbsns;   ///< The matrix Sbsns in the canonical master matrix.
    MatrixView Sbsp;    ///< The matrix Sbsp in the canonical master matrix.
    MatrixView Rbs;     ///< The matrix Rbs in the echelonizer matrix R = [Rbs; 0].
    IndicesView jb;     ///< The indices of the basic variables ordered as jb = (jbs).
    IndicesView jn;     ///< The indices of the non-basic variables ordered as jn = (jns, jnu).
    IndicesView js;     ///< The indices of the stable variables ordered as js = (jbs, jns).
    IndicesView ju;     ///< The indices of the unstable variables ordered as ju = (jbu, jnu).
};

} // namespace Optima

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
#include <Optima/MatrixRWQ.hpp>

namespace Optima {

/// Used to represent matrix *H = [Hxx Hpx]* in a master matrix.
struct MatrixViewH
{
    MatrixConstRef Hxx;   ///< The matrix *Hxx* in *H = [Hxx Hxp]*.
    MatrixConstRef Hxp;   ///< The matrix *Hxp* in *H = [Hxx Hxp]*.
    const bool isHxxDiag; ///< The flag that indicates wether *Hxx* is diagonal.
};

/// Used to represent matrix *V = [Vpx Vpp]* in a master matrix.
struct MatrixViewV
{
    MatrixConstRef Vpx; ///< The matrix *Vpx* in *V = [Vpx Vpp]*.
    MatrixConstRef Vpp; ///< The matrix *Vpp* in *V = [Vpx Vpp]*.
};

/// Used to represent matrix *W = [Wx Wp] = [Ax Ap; Jx Jp]* in a master matrix.
struct MatrixViewW
{
    MatrixConstRef Wx; ///< The matrix *Wx* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*.
    MatrixConstRef Wp; ///< The matrix *Wp* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*.
};

/// Used to represent a master matrix.
struct MasterMatrix
{
    MatrixViewH H;      ///< The matrix *H = [Hxx Hxp]* in the master matrix.
    MatrixViewV V;      ///< The matrix *V = [Vpx Vpp]* in the master matrix.
    MatrixViewW W;      ///< The matrix *W = [Ax Ap; Jx Jp]* in the master matrix.
    MatrixViewRWQ RWQ;  ///< The echelon form *RWQ = [Ibb Sbn Sbp]* of matrix *W* in the master matrix.
    IndicesConstRef js; ///< The indices of the stable variables in *x*.
    IndicesConstRef ju; ///< The indices of the unstable non-basic variables in *x*.

    /// Convert this MasterMatrixBase object into a Matrix object.
    operator Matrix() const;
};

} // namespace Optima

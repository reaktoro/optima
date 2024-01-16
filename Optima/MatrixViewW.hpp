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
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent a view to matrix *W = [Wx Wp] = [Ax Ap; Jx Jp]*.
struct MatrixViewW
{
    MatrixView Wx;  ///< The view to sub-matrix *Wx* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixView Wp;  ///< The view to sub-matrix *Wp* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixView Ax;  ///< The view to sub-matrix *Ax* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixView Ap;  ///< The view to sub-matrix *Ap* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixView Jx;  ///< The view to sub-matrix *Jx* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
    MatrixView Jp;  ///< The view to sub-matrix *Jp* in *W = [Wx Wp] = [Ax Ap; Jx Jp]*
};

} // namespace Optima

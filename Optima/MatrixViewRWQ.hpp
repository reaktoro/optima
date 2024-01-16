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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the echelon form *RWQ = [Ibb Sbn Sbp]* of matrix *W*.
struct MatrixViewRWQ
{
    MatrixView R;   ///< The echelonizer matrix of W so that *RWQ = [Ibb Sbn Sbp]* with *Q = (jb, jn)*.
    MatrixView Sbn; ///< The matrix *Sbn* in the echelon form of *W*.
    MatrixView Sbp; ///< The matrix *Sbp* in the echelon form of *W*.
    IndicesView jb; ///< The indices of the basic variables in the echelon form of *W*.
    IndicesView jn; ///< The indices of the non-basic variables in the echelon form of *W*.
};

} // namespace Optima

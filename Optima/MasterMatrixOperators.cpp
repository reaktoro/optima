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

#include "MasterMatrixOperators.hpp"

// Optima includes
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterVector.hpp>

namespace Optima {

auto tr(const MasterMatrix& M) -> MasterMatrixTr
{
    return { M };
}

auto operator*(const MasterMatrix& M, const MasterVector& u) -> MasterVector
{
    const auto nx = u.x.size();
    const auto np = u.p.size();
    const auto ny = u.y.size();
    const auto nz = u.z.size();

    MasterVector v(nx, np, ny, nz);

    v.data = M.matrix() * u.data; // TODO: Avoid M.matrix(), which is inefficient since it constructs a Matrix object.

    return v;
}

auto operator*(const MasterMatrixTr& Mtr, const MasterVector& u) -> MasterVector
{
    const auto nx = u.x.size();
    const auto np = u.p.size();
    const auto ny = u.y.size();
    const auto nz = u.z.size();

    MasterVector v(nx, np, ny, nz);

    const auto M = Mtr.M;

    v.data = M.matrix().transpose() * u.data; // TODO: Avoid M.matrix(), which is inefficient since it constructs a Matrix object.

    return v;
}

} // namespace Optima

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

#include "JacobianBlockV.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

JacobianBlockV::JacobianBlockV(Index nx, Index np)
: JacobianBlockV(zeros(np, nx), zeros(np, np))
{}

JacobianBlockV::JacobianBlockV(MatrixConstRef Vpx, MatrixConstRef Vpp)
: _Vpx(Vpx), _Vpp(Vpp), Vpx(_Vpx), Vpp(_Vpp)
{}

JacobianBlockV::JacobianBlockV(const JacobianBlockV& other)
: _Vpx(other._Vpx), _Vpp(other._Vpp), Vpx(_Vpx), Vpp(_Vpp)
{}

} // namespace Optima

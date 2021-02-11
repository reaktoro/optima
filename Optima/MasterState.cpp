// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

#include "MasterState.hpp"

namespace Optima {

MasterState::MasterState()
{}

MasterState::MasterState(const MasterDims& dims)
: u(dims),
  s(zeros(dims.nx))
//   xc(zeros(dims.nx, dims.nc)),
//   pc(zeros(dims.np, dims.nc)),
//   wc(zeros(dims.nw, dims.nc)),
//   sc(zeros(dims.nx, dims.nc))
{}

auto MasterState::resize(const MasterDims& dims) -> void
{
    u.resize(dims);
    s.resize(dims.nx);
    // xc.resize(dims.nx, dims.nc);
    // pc.resize(dims.np, dims.nc);
    // wc.resize(dims.nw, dims.nc);
    // sc.resize(dims.nx, dims.nc);
}

} // namespace Optima

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

#include "MasterMatrix.hpp"

namespace Optima {

MasterMatrix::operator Matrix() const
{
    using Eigen::all;
    const auto [nx, np, ny, nz, nw, nt] = dims;
    const auto ns = js.rows();
    const auto nu = ju.rows();
    Matrix M = zeros(nt, nt);
    auto Hxx = M.topRows(nx).leftCols(nx);
    auto Hxp = M.topRows(nx).middleCols(nx, np);
    auto WxT = M.topRows(nx).rightCols(nw);
    auto Vpx = M.middleRows(nx, np).leftCols(nx);
    auto Vpp = M.middleRows(nx, np).middleCols(nx, np);
    auto Wx  = M.bottomRows(nw).leftCols(nx);
    auto Wp  = M.bottomRows(nw).middleCols(nx, np);
    const auto Ws = W.Wx(all, js);
    Hxx(js, js) = H.Hxx(js, js);
    Hxx(ju, ju) = identity(nu, nu);
    Hxp(js, all) = H.Hxp(js, all);
    WxT(js, all) = tr(Ws);
    Vpx(all, js) = V.Vpx(all, js);
    Vpp = V.Vpp;
    Wx(all, js) = Ws;
    Wp = W.Wp;
    return M;
}

} // namespace Optima

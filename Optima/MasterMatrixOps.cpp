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

#include "MasterMatrixOps.hpp"

namespace Optima {

/// Return the product of a master matrix and a master vector.
auto operator*(const MasterMatrix& M, const MasterVectorView& u) -> MasterVector
{
    using Eigen::all;
    const auto [dims, H, V, W, RWQ, js, ju] = M;
    const auto us = u.x(js).eval();
    const auto uu = u.x(ju);
    const auto up = u.p;
    const auto uw = u.w;
    const auto Hss = H.Hxx(js, js);
    const auto Hsp = H.Hxp(js, all);
    const auto Vps = V.Vpx(all, js);
    const auto Vpp = V.Vpp;
    const auto Ws = W.Wx(all, js);
    const auto Wp = W.Wp;
    MasterVector a(u);
    auto as = a.x(js);
    auto au = a.x(ju);
    auto& ap = a.p;
    auto& aw = a.w;
    as = Hss*us + Hsp*up + tr(Ws)*uw;
    au.noalias() = uu;
    ap.noalias() = Vps*us + Vpp*up;
    aw.noalias() = Ws*us + Wp*up;
    return a;
}

/// Return the product of a master matrix transpose and a master vector.
auto operator*(const MasterMatrixTrExpr& trM, const MasterVectorView& u) -> MasterVector
{
    using Eigen::all;
    const auto [dims, H, V, W, RWQ, js, ju] = trM.M;
    const auto us = u.x(js).eval();
    const auto uu = u.x(ju);
    const auto up = u.p;
    const auto uw = u.w;
    const auto Hss = H.Hxx(js, js);
    const auto Hsp = H.Hxp(js, all);
    const auto Vps = V.Vpx(all, js);
    const auto Vpp = V.Vpp;
    const auto Ws = W.Wx(all, js);
    const auto Wp = W.Wp;
    MasterVector a(u);
    auto as = a.x(js);
    auto au = a.x(ju);
    auto& ap = a.p;
    auto& aw = a.w;
    as = tr(Hss)*us + tr(Vps)*up + tr(Ws)*uw;
    au.noalias() = uu;
    ap.noalias() = tr(Hsp)*us + tr(Vpp)*up + tr(Wp)*uw;
    aw.noalias() = Ws*us;
    return a;
}

} // namespace Optima

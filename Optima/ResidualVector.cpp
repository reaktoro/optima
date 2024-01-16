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

#include "ResidualVector.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct ResidualVector::Impl
{
    MasterDims dims; ///< The dimensions of the master variables.

    Index ns = 0;  ///< The number of stable variables
    Index nu = 0;  ///< The number of ustable variables
    Index nbs = 0; ///< The number of stable basic variables

    Vector ax;
    Vector aw;

    Vector ap;
    Vector asu;
    Vector awbs;
    Vector awstar;  ///< The workspace for auxiliary vector aw(star)
    Vector xsu;

    Impl()
    {
    }

    auto update(ResidualVectorUpdateArgs args) -> void
    {
        const auto [Mc, Wx, Wp, x, p, y, z, g, v, b, h] = args;

        const auto dims = Mc.dims;
        const auto nx = dims.nx;
        const auto np = dims.np;
        const auto ny = dims.ny;
        const auto nz = dims.nz;
        const auto nw = ny + nz;

        assert(x.size() == nx);
        assert(p.size() == np);
        assert(y.size() == ny);
        assert(z.size() == nz);
        assert(g.size() == nx);
        assert(v.size() == np);
        assert(b.size() == ny);
        assert(h.size() == nz);

        ns  = Mc.dims.ns;
        nu  = Mc.dims.nu;
        nbs = Mc.dims.nbs;

        const auto nns = Mc.dims.nns;

        const auto js = Mc.js;
        const auto ju = Mc.ju;

        const auto Ax = Wx.topRows(ny);
        const auto Ap = Wp.topRows(ny);
        const auto Jx = Wx.bottomRows(nz);
        const auto Jp = Wp.bottomRows(nz);

        const auto As = Ax(all, js);
        const auto Au = Ax(all, ju);

        const auto Js = Jx(all, js);

        const auto Rbs   = Mc.Rbs;
        const auto Sbsns = Mc.Sbsns;
        const auto Sbsp  = Mc.Sbsp;

        const auto gs = g(js);

        asu.resize(nx);
        auto as = asu.head(ns);
        auto au = asu.tail(nu);

        xsu.resize(nx);
        auto xs = xsu.head(ns);
        auto xu = xsu.tail(nu);

        aw.resize(nw);
        auto ay = aw.head(ny);
        auto az = aw.tail(nz);

        const auto xbs = xs.head(nbs);
        const auto xns = xs.tail(nns);

        xs = x(js);
        xu = x(ju);

        ax.resize(nx);
        ax(js).noalias() = -(gs + tr(As)*y + tr(Js)*z);
        ax(ju).fill(0.0);

        as = ax(js);
        au.fill(0.0);

        ay.noalias() = -(Ax*x + Ap*p - b);
        az.noalias() = -h;

        ap = -v;

        awstar.resize(nw);
        awstar.head(ny) = b - Au*xu;
        awstar.tail(nz) = Js*xs + Jp*p - h;

        awbs = multiplyMatrixVectorWithoutResidualRoundOffError(Rbs, awstar);
        awbs.noalias() -= xbs + Sbsns*xns + Sbsp*args.p;
    }

    auto masterVector() const -> MasterVectorView
    {
        return {ax, ap, aw};
    }

    auto canonicalVector() const -> CanonicalVectorView
    {
        const auto as = asu.head(ns);
        const auto au = asu.tail(nu);
        return {as, au, ap, awbs};
    }
};

ResidualVector::ResidualVector()
: pimpl(new Impl())
{}

ResidualVector::ResidualVector(const ResidualVector& other)
: pimpl(new Impl(*other.pimpl))
{}

ResidualVector::~ResidualVector()
{}

auto ResidualVector::operator=(ResidualVector other) -> ResidualVector&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ResidualVector::update(ResidualVectorUpdateArgs args) -> void
{
    pimpl->update(args);
}

auto ResidualVector::masterVector() const -> MasterVectorView
{
    return pimpl->masterVector();
}

auto ResidualVector::canonicalVector() const -> CanonicalVectorView
{
    return pimpl->canonicalVector();
}

} // namespace Optima

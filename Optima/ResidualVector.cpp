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
    const Index nx = 0;   ///< The number of variables x.
    const Index np = 0;   ///< The number of variables p.
    const Index ny = 0;   ///< The number of variables y.
    const Index nz = 0;   ///< The number of variables z.
    const Index nw = 0;   ///< The number of variables w = (y, z).

    CanonicalDims dims; ///< /// The dimension details of the Jacobian matrix and its canonical form.

    Vector ap;
    Vector asu;
    Vector awbs;
    Vector awstar;  ///< The workspace for auxiliary vector aw(star)
    Vector xsu;

    Impl(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz)
    {
        ap.resize(np);
        asu.resize(nx);
        awbs.resize(nw);
        awstar.resize(nw);
        xsu.resize(nx);
    }

    auto update(ResidualVectorUpdateArgs args) -> void
    {
        const auto [M, x, p, y, z, g, v, b, h] = args;

        assert(x.size() == nx);
        assert(p.size() == np);
        assert(y.size() == ny);
        assert(z.size() == nz);
        assert(g.size() == nx);
        assert(v.size() == np);
        assert(b.size() == ny);
        assert(h.size() == nz);

        const auto Mbar = M.canonicalForm();

        dims = Mbar.dims;

        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;

        const auto js = Mbar.js;
        const auto ju = Mbar.ju;

        const auto As = Mbar.As;
        const auto Au = Mbar.Au;
        const auto Js = Mbar.Js;
        const auto Jp = Mbar.Jp;

        const auto Rbs   = Mbar.R.topRows(nbs);
        const auto Sbsns = Mbar.Sbn.topLeftCorner(nbs, nns);
        const auto Sbsp  = Mbar.Sbp.topRows(nbs);

        const auto gs = g(js);

        auto as = asu.head(ns);
        auto au = asu.tail(nu);

        auto xs = xsu.head(ns);
        auto xu = xsu.tail(nu);

        const auto xbs = xs.head(nbs);
        const auto xns = xs.tail(nns);

        xs = x(js);
        xu = x(ju);

        as.noalias() = -(gs + tr(As)*y + tr(Js)*z);
        au.fill(0.0);

        ap = -v;

        awstar.head(ny) = b - Au*xu;
        awstar.tail(nz) = Js*xs + Jp*p - h;

        awbs = multiplyMatrixVectorWithoutResidualRoundOffError(Rbs, awstar);
        awbs.noalias() -= xbs + Sbsns*xns + Sbsp*args.p;
    }

    auto canonicalVector() const -> CanonicalVector
    {
        const auto ns = dims.ns;
        const auto as = asu.head(ns);
        return {as, ap, awbs};
    }
};

ResidualVector::ResidualVector(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
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

auto ResidualVector::canonicalVector() const -> CanonicalVector
{
    return pimpl->canonicalVector();
}

} // namespace Optima

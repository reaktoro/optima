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

#include "ResidualErrors.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct ResidualErrors::Impl
{
    const MasterDims dims; ///< The dimensions of the master variables.

    Vector ex;     ///< The residual errors associated with the first-order optimality conditions.
    Vector ep;     ///< The residual errors associated with the external constraint equations.
    Vector ew;     ///< The residual errors associated with the linear and non-linear constraint equations.
    double errorf; ///< The maximum residual error associated with the first-order optimality conditions.
    double errorv; ///< The maximum residual error associated with the external constraint equations.
    double errorw; ///< The maximum residual error associated with the linear and non-linear constraint equations.
    double error;  ///< The maximum error among all others.

    Impl(const MasterDims& dims)
    : dims(dims)
    {
        ex = zeros(dims.nx);
        ep = zeros(dims.np);
        ew = zeros(dims.nw);
    }

    auto update(MasterVectorView u, const ResidualFunction& F) -> void
    {
        const auto state = F.state();

        const auto Jc = state.Jc;
        const auto Fc = state.Fc;

        const auto nbs = Jc.dims.nbs;
        const auto nbl = Jc.dims.nl;

        const auto js = Jc.js;
        const auto ju = Jc.ju;

        const auto jbs = js.head(nbs);

        const auto rs   = Fc.xs;
        const auto rp   = Fc.p;
        const auto rwbs = Fc.wbs;

        const auto x = u.x;
        const auto gs = state.fx(js);
        const auto xbs = x(jbs);
        const auto xbslower = state.problem.xlower(jbs);
        const auto xbsupper = state.problem.xupper(jbs);

        ex(js) = abs(rs)/(1 + abs(gs));
        ex(ju).fill(0.0);

        auto ewbs = ew.head(nbs);
        auto ewbl = ew.tail(nbl);

        ewbs.noalias() = rwbs/((xbs.array() != 0.0).select(xbs, 1.0));
        ewbl.fill(0.0);

        ewbs.noalias() = (xbs.array() == xbslower.array()).select(0.0, ewbs);
        ewbs.noalias() = (xbs.array() == xbsupper.array()).select(0.0, ewbs);

        errorf = norminf(ex(js));
        errorv = norminf(rp);
        errorw = norminf(ewbs);
        error = std::max({ errorf, errorv, errorw });
    }
};

ResidualErrors::ResidualErrors(const MasterDims& dims)
: pimpl(new Impl(dims)),
  ex(pimpl->ex),
  ep(pimpl->ep),
  ew(pimpl->ew),
  errorf(pimpl->errorf),
  errorv(pimpl->errorv),
  errorw(pimpl->errorw),
  error(pimpl->error)
{}

ResidualErrors::ResidualErrors(const ResidualErrors& other)
: pimpl(new Impl(*other.pimpl)),
  ex(pimpl->ex),
  ep(pimpl->ep),
  ew(pimpl->ew),
  errorf(pimpl->errorf),
  errorv(pimpl->errorv),
  errorw(pimpl->errorw),
  error(pimpl->error)
{}

ResidualErrors::~ResidualErrors()
{}

auto ResidualErrors::operator=(ResidualErrors other) -> ResidualErrors&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ResidualErrors::update(MasterVectorView u, const ResidualFunction& F) -> void
{
    pimpl->update(u, F);
}

} // namespace Optima

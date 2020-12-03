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

#include "ResidualErrors.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct ResidualErrors::Impl
{
    const MasterDims dims; ///< The dimensions of the master variables.

    Vector xlower; ///< The lower bounds for variables *x*.
    Vector xupper; ///< The upper bounds for variables *x*.
    Vector plower; ///< The lower bounds for variables *p*.
    Vector pupper; ///< The upper bounds for variables *p*.

    Vector ex;     ///< The residual errors associated with the first-order optimality conditions.
    Vector ep;     ///< The residual errors associated with the external constraint equations.
    Vector ew;     ///< The residual errors associated with the linear and non-linear constraint equations.
    double errorx; ///< The maximum residual error associated with the first-order optimality conditions.
    double errorp; ///< The maximum residual error associated with the external constraint equations.
    double errorw; ///< The maximum residual error associated with the linear and non-linear constraint equations.
    double error;  ///< The error norm sqrt(||ex||^2 + ||ep||^2 + ||ew||^2).

    Impl(const MasterDims& dims)
    : dims(dims)
    {
        ex = zeros(dims.nx);
        ep = zeros(dims.np);
        ew = zeros(dims.nw);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        xlower = problem.xlower;
        xupper = problem.xupper;
        plower = problem.plower;
        pupper = problem.pupper;
    }

    auto update(MasterVectorView u, const ResidualFunction& F) -> void
    {
        sanitycheck();

        const auto res = F.result();

        const auto Jc = res.Jc;
        const auto Fc = res.Fc;

        const auto nbs = Jc.dims.nbs;
        const auto nbl = Jc.dims.nl;

        const auto js = Jc.js;
        const auto ju = Jc.ju;

        const auto jbs = js.head(nbs);

        const auto rs   = Fc.xs;
        const auto rp   = Fc.p;
        const auto rwbs = Fc.wbs;

        const auto x = u.x;
        const auto gs = res.f.fx(js);
        const auto xbs = x(jbs);
        const auto xbslower = xlower(jbs);
        const auto xbsupper = xupper(jbs);

        ex(js) = abs(rs);
        ex(ju).fill(0.0);

        ep = rp;

        auto ewbs = ew.head(nbs);
        auto ewbl = ew.tail(nbl);

        ewbs.noalias() = abs(rwbs);
        ewbl.fill(0.0);

        // Note: If a basic variable is attached to its lower/upper bound, this
        // means the last iteration tried to further decrease/increase the
        // variable. Ensure errors associated with such basic variables
        // attached to their bounds are zeroed out below.

        // Ensure basic variables on the bounds have zero optimality error
        ex(jbs) = (xbs.array() == xbslower.array()).select(0.0, ex(jbs));
        ex(jbs) = (xbs.array() == xbsupper.array()).select(0.0, ex(jbs));

        // Ensure basic variables on the bounds have zero feasibility error
        ewbs.noalias() = (xbs.array() == xbslower.array()).select(0.0, ewbs);
        ewbs.noalias() = (xbs.array() == xbsupper.array()).select(0.0, ewbs);

        errorx = norminf(ex(js));
        errorp = norminf(rp);
        errorw = norminf(ewbs);
        error = std::sqrt(ex.squaredNorm() + rp.squaredNorm() + ewbs.squaredNorm());
    }

    auto sanitycheck() const -> void
    {
        assert(dims.nx > 0);
        assert(ex.size() == dims.nx);
        assert(ep.size() == dims.np);
        assert(ew.size() == dims.nw);
        assert(xlower.size() == dims.nx);
        assert(xupper.size() == dims.nx);
        assert(plower.size() == dims.np);
        assert(pupper.size() == dims.np);
    }
};

ResidualErrors::ResidualErrors(const MasterDims& dims)
: pimpl(new Impl(dims)),
  ex(pimpl->ex),
  ep(pimpl->ep),
  ew(pimpl->ew),
  errorx(pimpl->errorx),
  errorp(pimpl->errorp),
  errorw(pimpl->errorw),
  error(pimpl->error)
{}

ResidualErrors::ResidualErrors(const ResidualErrors& other)
: pimpl(new Impl(*other.pimpl)),
  ex(pimpl->ex),
  ep(pimpl->ep),
  ew(pimpl->ew),
  errorx(pimpl->errorx),
  errorp(pimpl->errorp),
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

auto ResidualErrors::initialize(const MasterProblem& problem) -> void
{
    return pimpl->initialize(problem);
}

auto ResidualErrors::update(MasterVectorView u, const ResidualFunction& F) -> void
{
    pimpl->update(u, F);
}

} // namespace Optima

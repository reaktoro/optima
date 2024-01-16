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

#include "ResidualErrors.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct ResidualErrors::Impl
{
    MasterDims dims; ///< The dimensions of the master variables.

    Vector xlower; ///< The lower bounds for variables *x*.
    Vector xupper; ///< The upper bounds for variables *x*.
    Vector plower; ///< The lower bounds for variables *p*.
    Vector pupper; ///< The upper bounds for variables *p*.

    Vector ex; ///< The residual errors associated with the first-order optimality conditions.
    Vector ep; ///< The residual errors associated with the external constraint equations.
    Vector ew; ///< The residual errors associated with the linear and non-linear constraint equations.
    Vector ewbar; ///< The residual errors associated with the linear and non-linear constraint equations in canonical form.

    double errorx = 0.0; ///< The maximum residual error associated with the first-order optimality conditions.
    double errorp = 0.0; ///< The maximum residual error associated with the external constraint equations.
    double errorw = 0.0; ///< The maximum residual error associated with the linear and non-linear constraint equations in canonical form.
    double error  = 0.0; ///< The error norm sqrt(||ex||^2 + ||ep||^2 + ||ew||^2).

    Impl()
    {}

    auto initialize(const MasterProblem& problem) -> void
    {
        dims = problem.dims;
        xlower = problem.xlower;
        xupper = problem.xupper;
        plower = problem.plower;
        pupper = problem.pupper;
        ex = zeros(dims.nx);
        ep = zeros(dims.np);
        ew = zeros(dims.nw);
        ewbar = zeros(dims.nw);
    }

    auto update(MasterVectorView u, const ResidualFunction& F) -> void
    {
        sanitycheck();

        const auto res = F.result();

        const auto Jc = res.Jc;
        const auto Fc = res.Fc;
        const auto Fm = res.Fm;

        const auto nbs = Jc.dims.nbs;
        const auto nbl = Jc.dims.nl;

        const auto js = Jc.js;
        const auto ju = Jc.ju;

        const auto jbs = js.head(nbs);

        const auto x = u.x;
        const auto xbs = x(jbs);
        const auto xbslower = xlower(jbs);
        const auto xbsupper = xupper(jbs);

        ex = abs(Fm.x);
        ep = abs(Fm.p);
        ew = abs(Fm.w);

        auto ewbs = ewbar.head(nbs);
        auto ewbl = ewbar.tail(nbl);

        ewbs = abs(Fc.wbs);
        ewbl.fill(0.0);

        // Ensure currently unstable x variables have zero optimality errors.
        ex(ju).fill(0.0);

        // Note: If a basic variable is attached to its lower/upper bound, this
        // means the last iteration tried to further decrease/increase the
        // variable. Ensure optimality errors associated with such basic
        // variables attached to their bounds are zeroed out below.

        // Ensure basic variables on the bounds have zero optimality error
        ex(jbs) = (xbs.array() == xbslower.array()).select(0.0, ex(jbs));
        ex(jbs) = (xbs.array() == xbsupper.array()).select(0.0, ex(jbs));

        errorx = norminf(ex);
        errorp = norminf(ep);
        errorw = norminf(ewbar); // prefer error check at the canonical level
        error = std::max({errorx, errorp, errorw});
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

ResidualErrors::ResidualErrors()
: pimpl(new Impl())
{}

ResidualErrors::ResidualErrors(const ResidualErrors& other)
: pimpl(new Impl(*other.pimpl))
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

auto ResidualErrors::ex() const -> VectorView { return pimpl->ex; }
auto ResidualErrors::ep() const -> VectorView { return pimpl->ep; }
auto ResidualErrors::ew() const -> VectorView { return pimpl->ew; }
auto ResidualErrors::ewbar() const -> VectorView { return pimpl->ewbar; }

auto ResidualErrors::errorx() const -> double { return pimpl->errorx; }
auto ResidualErrors::errorp() const -> double { return pimpl->errorp; }
auto ResidualErrors::errorw() const -> double { return pimpl->errorw; }
auto ResidualErrors::error() const -> double { return pimpl->error; }

} // namespace Optima

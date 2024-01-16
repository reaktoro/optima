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

#include "LineSearch.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/MasterMatrixOps.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct LineSearch::Impl
{
    MasterDims dims;           ///< The dimensions of the master variables.
    MasterVector utrial;       ///< The trial state of u = (x, p, y, z) during the line search minimization.
    LineSearchOptions options; ///< The options for the line search minimization.
    Vector xlower;             ///< The lower bounds for x.
    Vector xupper;             ///< The upper bounds for x.
    Vector plower;             ///< The lower bounds for p.
    Vector pupper;             ///< The upper bounds for p.
    MasterVector du;           ///< The Newton step du = u - uo
    MasterVector Jdu;          ///< The multiplication J * du

    Impl()
    {}

    auto setOptions(const LineSearchOptions opts) -> void
    {
        options = opts;
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        dims   = problem.dims;
        xlower = problem.xlower;
        xupper = problem.xupper;
        plower = problem.plower;
        pupper = problem.pupper;
        utrial.resize(dims);
    }

    auto isDescentDirection(MasterVectorView uo, MasterVectorView u, const ResidualFunction& F) -> bool
    {
        const auto res = F.result();
        const auto Fm = res.Fm;
        const auto Jm = res.Jm;
        du = u - uo;
        Jdu = Jm * du;
        const auto slope = Fm.dot(Jdu);
        return slope < 0.0;
    }

    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {
        warningif(!isDescentDirection(uo, u, F), "Proceeding with linear-search algorithm even though current Newton step is not a descent direction.");

        assert((u.x.array() <= xupper.array()).all());
        assert((u.x.array() >= xlower.array()).all());

        auto phi = [&](auto alpha)
        {
            utrial = uo*(1 - alpha) + alpha*u;
            F.update(utrial);
            E.update(utrial, F);
            return E.error();
        };

        const auto tol = options.tolerance;
        const auto maxiters = options.maxiterations;

        // Minimize phi(alpha) along the path from uo to u for alpha in [0, 1].
        const auto alphamin = minimizeBrent(phi, 0.0, 1.0, tol, maxiters);

        u = uo*(1 - alphamin) + alphamin*u; // using uo + alpha*(u - uo) is sensitive to round-off errors!

        F.update(u);
        E.update(u, F);

        // TODO: Consider instead setting u = utrial, since utrial has been
        // updated inside phi calls, and its last update correspond to uo*(1 -
        // alphamin) + alphamin*u. Also consider not updating F and E above in
        // this case, which have also been done in the last phi call.
    }
};

LineSearch::LineSearch()
: pimpl(new Impl())
{}

LineSearch::LineSearch(const LineSearch& other)
: pimpl(new Impl(*other.pimpl))
{}

LineSearch::~LineSearch()
{}

auto LineSearch::operator=(LineSearch other) -> LineSearch&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto LineSearch::setOptions(const LineSearchOptions& options) -> void
{
    pimpl->setOptions(options);
}

auto LineSearch::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto LineSearch::execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
{
    pimpl->execute(uo, u, F, E);
}

} // namespace Optima

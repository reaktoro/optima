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

#include "LineSearch.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct LineSearch::Impl
{
    /// The trial state of u = (x, p, y, z) during the line search minimization.
    MasterVector utrial;

    /// The options for the line search minimization.
    LineSearchOptions options;

    Impl(const MasterDims& dims)
    : utrial(dims)
    {
    }

    auto setOptions(const LineSearchOptions opts) -> void
    {
        options = opts;
    }

    auto start(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {
        auto phi = [&](auto alpha)
        {
            utrial = uo*(1 - alpha) + alpha*u;
            F.update(utrial);
            E.update(utrial, F);
            return E.error;
        };

        const auto tol = options.tolerance;
        const auto maxiters = options.maxiterations;

        // Minimize phi(alpha) along the path from uo to u for alpha in [0, 1].
        const auto alphamin = minimizeBrent(phi, 0.0, 1.0, tol, maxiters);

        u = uo*(1 - alphamin) + alphamin*u; // using uo + alpha*(u - uo) is sensitive to round-off errors!
    }
};

LineSearch::LineSearch(const MasterDims& dims)
: pimpl(new Impl(dims))
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

auto LineSearch::start(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
{
    pimpl->start(uo, u, F, E);
}

} // namespace Optima

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

#include "Convergence.hpp"

// C++ includes
#include <vector>

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct Convergence::Impl
{
    /// The collected errors for the convergence analysis.
    std::vector<double> history;

    /// The options for convergence analysis.
    ConvergenceOptions options;

    Impl()
    {
    }

    auto initialize(const MasterProblem& problem) -> void
    {
    }

    auto update(const ResidualErrors& E) -> void
    {
        history.push_back(E.error());
    }

    auto converged(ConvergenceCheckArgs const& args) const -> bool
    {
        if(history.empty()) return false;
        const auto currenterror = history.back();
        if(currenterror < options.tolerance)
            return true;
        return options.check && options.check(args);
    }

    auto rate() const -> double
    {
        const auto N = history.size();
        if(N < 3)
            return 0.0;
        const auto E1 = history[N - 1];
        const auto E2 = history[N - 2];
        const auto E3 = history[N - 3];
        return (E2 - E1)/(E3 - E2);
    }
};

Convergence::Convergence()
: pimpl(new Impl())
{}

Convergence::Convergence(const Convergence& other)
: pimpl(new Impl(*other.pimpl))
{}

Convergence::~Convergence()
{}

auto Convergence::operator=(Convergence other) -> Convergence&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Convergence::setOptions(const ConvergenceOptions& options) -> void
{
    pimpl->options = options;
}

auto Convergence::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto Convergence::update(const ResidualErrors& E) -> void
{
    pimpl->update(E);
}

auto Convergence::converged(ConvergenceCheckArgs const& args) const -> bool
{
    return pimpl->converged(args);
}

auto Convergence::rate() const -> double
{
    return pimpl->rate();
}

} // namespace Optima

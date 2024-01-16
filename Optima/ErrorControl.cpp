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

#include "ErrorControl.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/BacktrackSearch.hpp>
#include <Optima/ErrorStatus.hpp>
#include <Optima/LineSearch.hpp>

namespace Optima {

struct ErrorControl::Impl
{
    ErrorStatus errorstatus;         ///< The current error status of the calculation.
    BacktrackSearch backtracksearch; ///< The backtrack algorithm to correct steps producing infinity errors.
    LineSearch linesearch;           ///< The line-search algorithm to correct steps producing significant large errors.

    Impl()
    {}

    auto setOptions(const ErrorControlOptions& options) -> void
    {
        errorstatus.setOptions(options.errorstatus);
        backtracksearch.setOptions(options.backtracksearch);
        linesearch.setOptions(options.linesearch);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        errorstatus.initialize();
        backtracksearch.initialize(problem);
        linesearch.initialize(problem);
    }

    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {
        const auto error_prev = E.error();

        backtracksearch.execute(uo, u, F, E);

        const auto error_new = E.error();

        // if(error_new >= error_prev)
        //     linesearch.execute(uo, u, F, E);
    }
};

ErrorControl::ErrorControl()
: pimpl(new Impl())
{}

ErrorControl::ErrorControl(const ErrorControl& other)
: pimpl(new Impl(*other.pimpl))
{}

ErrorControl::~ErrorControl()
{}

auto ErrorControl::operator=(ErrorControl other) -> ErrorControl&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ErrorControl::setOptions(const ErrorControlOptions& options) -> void
{
    pimpl->setOptions(options);
}

auto ErrorControl::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto ErrorControl::execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
{
    pimpl->execute(uo, u, F, E);
}

} // namespace Optima

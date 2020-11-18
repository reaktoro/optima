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

#include "ErrorControl.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/BacktrackSearch.hpp>
#include <Optima/LineSearch.hpp>

namespace Optima {

struct ErrorControl::Impl
{
    /// The backtrack algorithm to correct steps producing infinity errors.
    BacktrackSearch backtracksearch;

    /// The line-search algorithm to correct steps producing significant large errors.
    LineSearch linesearch;


    Impl(const MasterDims& dims)
    : backtracksearch(dims), linesearch(dims)
    {
    }

    auto initialize(const MasterProblem& problem) -> void
    {
    }

    auto isBacktrackSearchNeeded(const ResidualErrors& E)
    {
        // return E.errorIsntFinite();
    }

    auto isLineSearchNeeded(const ResidualErrors& E)
    {
        // return E.errorHasIncreasedSignificantly();
    }

    auto executeBacktrackSearch(MasterVectorConstRef uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {

    }

    auto executeLineSearch(MasterVectorConstRef uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {

    }

    auto execute(MasterVectorConstRef uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
    {
        // if(E.errorIsInf()) {
        //     backtracksearch.start(F, E, uo, u);
        // }

        // if(E.errorHasIncreasedSignificantly()) {
        //     linesearch.start(F, E, uo, u);
        // }
    }
};

ErrorControl::ErrorControl(const MasterDims& dims)
: pimpl(new Impl(dims))
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

auto ErrorControl::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto ErrorControl::execute(MasterVectorConstRef uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void
{
    pimpl->execute(uo, u, F, E);
}

} // namespace Optima

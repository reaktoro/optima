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

#include "LineSearch.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct LineSearch::Impl
{
    Impl(const MasterDims& dims)
    {
    }

    auto start(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ErrorStatus& E) -> void
    {

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

auto LineSearch::start(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ErrorStatus& E) -> void
{

}

} // namespace Optima

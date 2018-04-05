// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include "Partition.hpp"

// Optima includes
#include <Optima/IndexUtils.hpp>

namespace Optima {

Partition::Partition()
: _nf(0)
{}

Partition::Partition(Index n)
: _ordering(indices(n)), _nf(0)
{}

auto Partition::setFixedVariables(IndicesConstRef inds) -> void
{
    const Index n = numVariables();
    _nf = inds.size();
    _ordering.noalias() = indices(n);
    _ordering.tail(_nf).swap(_ordering(inds));
}

auto Partition::setFreeVariables(IndicesConstRef inds) -> void
{
    const Index n = numVariables();
    const Index nx = inds.size();
    _nf = n - nx;
    _ordering.noalias() = indices(n);
    _ordering.head(nx).swap(_ordering(inds));
}

auto Partition::numVariables() const -> Index
{
    return ordering().size();
}

auto Partition::numFreeVariables() const -> Index
{
    return numVariables() - numFixedVariables();
}

auto Partition::numFixedVariables() const -> Index
{
    return _nf;
}

auto Partition::freeVariables() const -> IndicesConstRef
{
    return ordering().head(numFreeVariables());
}

auto Partition::fixedVariables() const -> IndicesConstRef
{
    return ordering().tail(numFixedVariables());
}

auto Partition::ordering() const -> IndicesConstRef
{
    return _ordering;
}

} // namespace Optima

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

#include "ObjectiveFunction.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

ObjectiveFunction::ObjectiveFunction()
{
    fn = [](ObjectiveResult& res, VectorConstRef x, VectorConstRef p, ObjectiveOptions opts)
    {
        error(true, "The objective function has not been initialized.");
        return false;
    };
}

ObjectiveFunction::ObjectiveFunction(const Signature& func)
{
    error(func == nullptr, "ObjectiveFunction cannot be constructed with a non-initialized function.");
    fn = func;
}

ObjectiveFunction::ObjectiveFunction(const Signature4py& func)
{
    error(func == nullptr, "ObjectiveFunction cannot be constructed with a non-initialized function.");
    fn = [=](ObjectiveResult& res, VectorConstRef x, VectorConstRef p, ObjectiveOptions opts)
    {
        return func(&res, x, p, opts);
    };
}

auto ObjectiveFunction::operator()(ObjectiveResult& res, VectorConstRef x, VectorConstRef p, ObjectiveOptions opts) const -> bool
{
    return fn(res, x, p, opts);
}

} // namespace Optima

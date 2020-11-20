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
    fn = [](ObjectiveResult& res, VectorView x, VectorView p, ObjectiveOptions opts)
    {
        error(true, "Cannot evaluate a non-initialized ObjectiveFunction object");
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
    fn = [=](ObjectiveResult& res, VectorView x, VectorView p, ObjectiveOptions opts)
    {
        func(&res, x, p, opts);
    };
}

auto ObjectiveFunction::operator()(ObjectiveResult& res, VectorView x, VectorView p, ObjectiveOptions opts) const -> void
{
    // Ensure default status for some ObjectiveResult members before evaluation
    res.diagfxx = false;
    res.fxx4basicvars = false;
    res.succeeded = true;
    fn(res, x, p, opts);
}

} // namespace Optima

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

#include "ObjectiveFunction.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

ObjectiveFunction::ObjectiveFunction()
{
    fn = [](ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts)
    {
        // initialized with a "do nothing function"!
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
    fn = [=](ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts)
    {
        func(&res, x, p, c, opts);
    };
}

auto ObjectiveFunction::operator()(ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts) const -> void
{
    // Ensure clear state before evaluation
    res.f = 0.0;
    res.fx.fill(0.0);
    res.fxx.fill(0.0);
    res.fxp.fill(0.0);
    res.fxc.fill(0.0);
    res.diagfxx = false;
    res.fxx4basicvars = false;
    res.succeeded = true;
    fn(res, x, p, c, opts);
}

auto ObjectiveFunction::operator=(const Signature& func) -> ObjectiveFunction&
{
    error(func == nullptr, "ObjectiveFunction cannot be constructed with a non-initialized function.");
    fn = func;
    return *this;
}

auto ObjectiveFunction::initialized() const -> bool
{
    return fn != nullptr;
}

} // namespace Optima

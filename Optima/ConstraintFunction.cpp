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

#include "ConstraintFunction.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

ConstraintFunction::ConstraintFunction()
{
    fn = [](ConstraintResultRef res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts)
    {
        // initialized with a "do nothing function"!
    };
}

ConstraintFunction::ConstraintFunction(const Signature& func)
{
    error(func == nullptr, "ConstraintFunction cannot be constructed with a non-initialized function.");
    fn = func;
}

ConstraintFunction::ConstraintFunction(const Signature4py& func)
{
    error(func == nullptr, "ConstraintFunction cannot be constructed with a non-initialized function.");
    fn = [=](ConstraintResultRef res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts)
    {
        func(&res, x, p, c, opts);
    };
}

auto ConstraintFunction::operator()(ConstraintResultRef res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts) const -> void
{
    // Ensure clear state before evaluation
    res.val.fill(0.0);
    res.ddx.fill(0.0);
    res.ddp.fill(0.0);
    res.ddc.fill(0.0);
    res.ddx4basicvars = false;
    res.succeeded = true;
    fn(res, x, p, c, opts);
}

auto ConstraintFunction::operator=(const Signature& func) -> ConstraintFunction&
{
    error(func == nullptr, "ConstraintFunction cannot be constructed with a non-initialized function.");
    fn = func;
    return *this;
}

auto ConstraintFunction::initialized() const -> bool
{
    return fn != nullptr;
}

} // namespace Optima

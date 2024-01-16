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

#include "ResourcesFunction.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

ResourcesFunction::ResourcesFunction()
{
    fn = [](VectorView x, VectorView p, VectorView c, ObjectiveOptions fopts, ConstraintOptions hopts, ConstraintOptions vopts)
    {
        // initialized with a "do nothing function"!
    };
}

ResourcesFunction::ResourcesFunction(const Signature& func)
{
    error(func == nullptr, "ResourcesFunction cannot be constructed with a non-initialized function.");
    fn = func;
}

auto ResourcesFunction::operator()(VectorView x, VectorView p, VectorView c, ObjectiveOptions fopts, ConstraintOptions hopts, ConstraintOptions vopts) const -> void
{
    fn(x, p, c, fopts, hopts, vopts);
}

auto ResourcesFunction::operator=(const Signature& func) -> ResourcesFunction&
{
    error(func == nullptr, "ResourcesFunction cannot be constructed with a non-initialized function.");
    fn = func;
    return *this;
}

auto ResourcesFunction::initialized() const -> bool
{
    return fn != nullptr;
}

} // namespace Optima

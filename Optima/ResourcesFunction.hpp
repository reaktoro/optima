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

#pragma once

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/ConstraintFunction.hpp>
#include <Optima/ObjectiveFunction.hpp>

namespace Optima {

/// Used to represent a function that precomputes internally shared resources for objective and constraint functions.
/// @see ConstraintFunction, ObjectiveFunction
class ResourcesFunction
{
public:
    /// The main functional signature of an objective function *f(x, p, c)*.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param fopts The options that will be transmitted to the evaluation of *f(x, p, c)*.
    /// @param hopts The options that will be transmitted to the evaluation of constraints functions *h(x, p, c)*.
    /// @param vopts The options that will be transmitted to the evaluation of constraints functions *v(x, p, c)*.
    using Signature = std::function<void(VectorView x, VectorView p, VectorView c, ObjectiveOptions fopts, ConstraintOptions hopts, ConstraintOptions vopts)>;

    /// Construct a default ResourcesFunction object.
    ResourcesFunction();

    /// Construct an ResourcesFunction object with given function.
    ResourcesFunction(const Signature& fn);

    // template<typename Func, EnableIf<!isStdFunction<Func>>...>
    // ResourcesFunction(const Func& fn)
    // : ResourcesFunction(fn) {}

    /// Evaluate the resources function.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param fopts The options that will be transmitted to the evaluation of *f(x, p, c)*.
    /// @param hopts The options that will be transmitted to the evaluation of constraints functions *h(x, p, c)*.
    /// @param vopts The options that will be transmitted to the evaluation of constraints functions *v(x, p, c)*.
    auto operator()(VectorView x, VectorView p, VectorView c, ObjectiveOptions fopts, ConstraintOptions hopts, ConstraintOptions vopts) const -> void;

    /// Assign another resources function to this.
    auto operator=(const Signature& fn) -> ResourcesFunction&;

    /// Return `true` if this ResourcesFunction object has been initialized.
    auto initialized() const -> bool;

private:
    /// The resources function with main functional signature.
    Signature fn;
};

} // namespace Optima

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

#pragma once

// C++ includes
#include <memory>

// Optima includes
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/CanonicalVector.hpp>
#include <Optima/Constants.hpp>
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterProblem.hpp>
#include <Optima/MasterVector.hpp>

namespace Optima {

// Forward declarations
struct ObjectiveResult;
struct ConstraintResult;

/// Used to determine whether the evaluation of objective and constraint functions succeeded or not.
struct ResidualFunctionUpdateStatus
{
    bool f = FAILED; ///< The flag indicating whether *f(x, p)* succeeded.
    bool h = FAILED; ///< The flag indicating whether *h(x, p)* succeeded.
    bool v = FAILED; ///< The flag indicating whether *v(x, p)* succeeded.
    operator bool() const { return f && h && v; }
};

/// The result of the residual function evaluation at *u = (x, p, y, z)*.
struct ResidualFunctionResult
{
    ObjectiveResult const& f;   ///< The evaluated result of the objective function *f(x, p)*.
    ConstraintResult const& h;  ///< The evaluated result of the constraint function *h(x, p)*.
    ConstraintResult const& v;  ///< The evaluated result of the constraint function *v(x, p)*.
};

/// Used to represent the residual function *F(u)* in the Newton step problem.
class ResidualFunction
{
public:
    /// Construct a ResidualFunction object.
    ResidualFunction(const MasterDims& dims);

    /// Construct a copy of a ResidualFunction object.
    ResidualFunction(const ResidualFunction& other);

    /// Destroy this ResidualFunction object.
    virtual ~ResidualFunction();

    /// Assign a ResidualFunction object to this.
    auto operator=(ResidualFunction other) -> ResidualFunction&;

    /// Initialize the residual function once before update computations.
    auto initialize(const MasterProblem& problem) -> void;

    /// Update the residual function with given *u = (x, p, y, z)*.
    auto update(MasterVectorView u) -> ResidualFunctionUpdateStatus;

    /// Update the residual function with given *u = (x, p, y, z)* skipping Jacobian evaluations.
    auto updateSkipJacobian(MasterVectorView u) -> ResidualFunctionUpdateStatus;

    /// Return the Jacobian matrix in canonical form.
    auto canonicalJacobianMatrix() const -> CanonicalMatrixView;

    /// Return the residual vector in canonical form.
    auto canonicalResidualVector() const -> CanonicalVectorView;

    /// Return the Jacobian matrix in master form.
    auto masterJacobianMatrix() const -> MasterMatrix;

    /// Return the residual vector in master form.
    auto masterResidualVector() const -> MasterVectorView;

    /// Return the result of the evaluation of the residual function.
    auto result() const -> ResidualFunctionResult;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

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

/// Used to determine whether the evaluation of objective and constraint functions succeeded or not.
struct ResidualFunctionUpdateStatus
{
    bool f = FAILED; ///< The flag indicating whether *f(x, p)* succeeded.
    bool h = FAILED; ///< The flag indicating whether *h(x, p)* succeeded.
    bool v = FAILED; ///< The flag indicating whether *v(x, p)* succeeded.
    operator bool() const { return f && h && v; }
};

/// Used to store the residual errors associated with the optimization calculation.
struct ResidualFunctionState
{
    const double f;           ///< The evaluated value of f(x, p).
    const VectorConstRef fx;  ///< The evaluated gradient of f(x, p) with respect to x.
    const MatrixConstRef fxx; ///< The evaluated Jacobian of fx(x, p) with respect to x.
    const MatrixConstRef fxp; ///< The evaluated Jacobian of fx(x, p) with respect to p.
    const bool diagfxx;       ///< The flag indicating whether `fxx` is diagonal.

    const VectorConstRef v;   ///< The evaluated value of v(x, p).
    const MatrixConstRef vx;  ///< The evaluated Jacobian of v(x, p) with respect to x.
    const MatrixConstRef vp;  ///< The evaluated Jacobian of v(x, p) with respect to p.

    const VectorConstRef h;   ///< The evaluated value of h(x, p).
    const MatrixConstRef hx;  ///< The evaluated Jacobian of h(x, p) with respect to x.
    const MatrixConstRef hp;  ///< The evaluated Jacobian of h(x, p) with respect to p.

    const MasterMatrix Jm;        /// The Jacobian matrix in master form.
    const CanonicalMatrixView Jc; /// The Jacobian matrix in canonical form.

    const MasterVectorView Fm;    /// The residual vector in master form.
    const CanonicalVectorView Fc; /// The residual vector in canonical form.
};

/// Used to represent the residual function *F(u)* in the Newton step problem.
class ResidualFunction
{
public:
    /// Construct a ResidualFunction object.
    /// @param dims The dimensions of the master variables
    /// @param Ax The matrix *Ax* in *W = [Ax Ap; Jx Jp]*.
    /// @param Ap The matrix *Ap* in *W = [Ax Ap; Jx Jp]*.
    ResidualFunction(const MasterDims& dims, MatrixConstRef Ax, MatrixConstRef Ap);

    /// Construct a copy of a ResidualFunction object.
    ResidualFunction(const ResidualFunction& other);

    /// Destroy this ResidualFunction object.
    virtual ~ResidualFunction();

    /// Assign a ResidualFunction object to this.
    auto operator=(ResidualFunction other) -> ResidualFunction&;

    /// Initialize this ResidualFunction object.
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

    /// Return the current state details of the residual function.
    auto state() const -> ResidualFunctionState;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

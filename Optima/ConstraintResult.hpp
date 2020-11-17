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
#include <Optima/MasterDims.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The result of a constraint function evaluation.
/// @see ConstraintFunction
class ConstraintResult
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// The evaluated vector value of *c(x, p)*.
    VectorRef val;

    /// The evaluated Jacobian matrix of *c(x, p)* with respect to *x*.
    MatrixRef ddx;

    /// The evaluated Jacobian matrix of *c(x, p)* with respect to *p*.
    MatrixRef ddp;

    /// True if `ddx` is non-zero only on columns corresponding to basic varibles in *x*.
    bool ddx4basicvars = false;

    /// Construct a ConstraintResult object.
    /// @param nc The number of constraint equations in *c(x, p)*.
    /// @param nx The number of variables in *x*.
    /// @param np The number of variables in *p*.
    ConstraintResult(Index nc, Index nx, Index np);

    /// Construct a copy of a ConstraintResult object.
    ConstraintResult(const ConstraintResult& other);

    /// Destroy this ConstraintResult object.
    virtual ~ConstraintResult();

    /// Assign a ConstraintResult object to this.
    auto operator=(ConstraintResult other) -> ConstraintResult& = delete;
};

} // namespace Optima

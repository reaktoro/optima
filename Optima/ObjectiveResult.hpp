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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The result of an objective function evaluation.
/// @see ObjectiveFunction
class ObjectiveResult
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// The evaluated objective function *f(x, p)*.
    double f = 0.0;

    /// The evaluated gradient vector of *f(x, p)* with respect to *x*.
    VectorRef fx;

    /// The evaluated Jacobian matrix of *fx(x, p)* with respect to *x*.
    MatrixRef fxx;

    /// The evaluated Jacobian matrix of *fx(x, p)* with respect to *p*.
    MatrixRef fxp;

    /// True if `fxx` is diagonal.
    bool diagfxx = false;

    /// True if `fxx` is non-zero only on columns corresponding to basic varibles in *x*.
    bool fxx4basicvars = false;

    /// Construct a ObjectiveResult object.
    /// @param nx The number of variables in *x*.
    /// @param np The number of variables in *p*.
    ObjectiveResult(Index nx, Index np);

    /// Construct a copy of a ObjectiveResult object.
    ObjectiveResult(const ObjectiveResult& other);

    /// Destroy this ObjectiveResult object.
    virtual ~ObjectiveResult();

    /// Assign a ObjectiveResult object to this.
    auto operator=(ObjectiveResult other) -> ObjectiveResult& = delete;
};

} // namespace Optima

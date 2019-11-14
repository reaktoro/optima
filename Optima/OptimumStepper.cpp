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

#include "OptimumStepper.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/IpSaddlePointSolver.hpp>
#include <Optima/Objective.hpp>
#include <Optima/OptimumOptions.hpp>
#include <Optima/OptimumParams.hpp>
#include <Optima/OptimumState.hpp>
#include <Optima/OptimumStructure.hpp>
#include <Optima/Result.hpp>
#include <Optima/VariantMatrix.hpp>

namespace Optima {

struct OptimumStepper::Impl
{
    /// The structure of the optimization calculation
    OptimumStructure structure;

    /// The options for the optimization calculation
    OptimumOptions options;

    /// The solution vector `s = [dx dy dz dw]`.
    Vector s;

    /// The right-hand side residual vector `r = [rx ry rz rw]`.
    Vector r;

    /// The matrices Z, W, L, U assuming the ordering x = [x(free) x(fixed)].
    Vector Z, W, L, U;

    /// The number of variables.
    Index n;

    /// The number of free and fixed variables.
    Index nx, nf;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// The interior-point saddle point solver.
    IpSaddlePointSolver solver;

    /// Construct a OptimumStepper::Impl instance with given optimization problem structure.
    Impl(const OptimumStructure& structure)
    : structure(structure)
    {
        // Initialize the members related to number of variables and constraints
        n  = structure.numVariables();
        m  = structure.numEqualityConstraints();
        nf = structure.variablesWithFixedValues().size();
        nx = n - nf;
        t  = 3*n + m;

        // Initialize Z and W with zeros (the dafault value for variables
        // with fixed values or no lower/upper bounds).
        Z = zeros(n);
        W = zeros(n);

        // Initialize L and U with ones (the dafault value for variables
        // with fixed values or no lower/upper bounds).
        L = ones(n);
        U = ones(n);

        // Initialize r and s with zeros.
        r = zeros(t);
        s = zeros(t);

        // Initialize the saddle point solver
        solver.initialize(structure.A);
    }

    /// Decompose the interior-point saddle point matrix for diagonal Hessian matrices.
    auto decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> Result
    {
        // The result of this method call
        Result res;

        // The indices of the variables with lower and upper bounds and fixed values
        IndicesConstRef ilower = structure.variablesWithLowerBounds();
        IndicesConstRef iupper = structure.variablesWithUpperBounds();
        IndicesConstRef ifixed = structure.variablesWithFixedValues();

        // Update Z and L for the variables with lower bounds
        Z(ilower) = state.z(ilower);
		L(ilower) = state.x(ilower) - params.xlower;

        // Update W and U for the variables with upper bounds
        W(iupper) = state.w(iupper);
        U(iupper) = state.x(iupper) - params.xupper;

        // Ensure entries in L are positive in case x[i] == lowerbound[i]
		for(Index i : ilower) L[i] = L[i] > 0.0 ? L[i] : options.mu;

        // Ensure entries in U are negative in case x[i] == upperbound[i]
		for(Index i : iupper) U[i] = U[i] < 0.0 ? U[i] : -options.mu;

        // Define the interior-point saddle point matrix
        IpSaddlePointMatrix spm(f.hessian, structure.A, Z, W, L, U, ifixed);

        // Decompose the interior-point saddle point matrix
        solver.decompose(spm);

        return res.stop();
    }

    /// Solve the interior-point saddle point matrix.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> Result
    {
        // The result of this method call
        Result res;

        // Alias to state variables
        VectorConstRef x = state.x;
        VectorConstRef y = state.y;
        VectorConstRef z = state.z;
        VectorConstRef w = state.w;

        // Alias to objective variables
        VectorConstRef g = f.gradient;

        // Alias to structure variables
        MatrixConstRef A = structure.A;

        // The indices of the variables with lower and upper bounds and fixed values
        IndicesConstRef ilower = structure.variablesWithLowerBounds();
        IndicesConstRef iupper = structure.variablesWithUpperBounds();
        IndicesConstRef ifixed = structure.variablesWithFixedValues();

        // Views to the sub-vectors in r = [a b c d]
        auto a = r.head(n);
        auto b = r.segment(n, m);
        auto c = r.segment(n + m, n);
        auto d = r.tail(n);

        // Calculate the optimality residual vector a
        a.noalias() = -(g + tr(A) * y - z - w);

        // Set a to zero for fixed variables
        a(ifixed).fill(0.0);

        // Calculate the feasibility residual vector b
        b.noalias() = -(A * x - params.b);

        // Calculate the centrality residual vectors c and d
        for(Index i : ilower) c[i] = options.mu - L[i] * z[i]; // TODO Check if mu is still needed. Maybe this algorithm no longer needs perturbation.
        for(Index i : iupper) d[i] = options.mu - U[i] * w[i];

//        c.fill(0.0); // TODO For example, there is no mu here and this seems to work
//        d.fill(0.0);

        // The right-hand side vector of the interior-point saddle point problem
        IpSaddlePointVector rhs(r, n, m);

        // The solution vector of the interior-point saddle point problem
        IpSaddlePointSolution sol(s, n, m);

        // Solve the saddle point problem
        solver.solve(rhs, sol);

        return res.stop();
    }

    /// Return the calculated Newton step vector.
    auto step() const -> IpSaddlePointVector
    {
        return IpSaddlePointVector(s, n, m);
    }

    /// Return the calculated residual vector for the current optimum state.
    auto residual() const -> IpSaddlePointVector
    {
        return IpSaddlePointVector(r, n, m);
    }

    /// Return the assembled interior-point saddle point matrix.
    auto matrix(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> IpSaddlePointMatrix
    {
        // The indices of the variables with fixed values
        IndicesConstRef ifixed = structure.variablesWithFixedValues();

        // Define the interior-point saddle point matrix
        return IpSaddlePointMatrix(f.hessian, structure.A, Z, W, L, U, ifixed);
    }
};

OptimumStepper::OptimumStepper(const OptimumStructure& structure)
: pimpl(new Impl(structure))
{}

OptimumStepper::OptimumStepper(const OptimumStepper& other)
: pimpl(new Impl(*other.pimpl))
{}

OptimumStepper::~OptimumStepper()
{}

auto OptimumStepper::operator=(OptimumStepper other) -> OptimumStepper&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto OptimumStepper::setOptions(const OptimumOptions& options) -> void
{
    pimpl->options = options;
    pimpl->solver.setOptions(options.kkt);
}

auto OptimumStepper::decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> Result
{
    return pimpl->decompose(params, state, f);
}

auto OptimumStepper::solve(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> Result
{
    return pimpl->solve(params, state, f);
}

auto OptimumStepper::matrix(const OptimumParams& params, const OptimumState& state, const ObjectiveResult& f) -> IpSaddlePointMatrix
{
    return pimpl->matrix(params, state, f);
}

auto OptimumStepper::step() const -> IpSaddlePointVector
{
    return pimpl->step();
}

auto OptimumStepper::residual() const -> IpSaddlePointVector
{
    return pimpl->residual();
}

} // namespace Optima

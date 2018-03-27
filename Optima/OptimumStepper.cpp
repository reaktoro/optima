// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <Optima/OptimumOptions.hpp>
#include <Optima/OptimumParams.hpp>
#include <Optima/OptimumState.hpp>
#include <Optima/OptimumStructure.hpp>
#include <Optima/Result.hpp>
#include <Optima/VariantMatrix.hpp>
using namespace Eigen;
using Eigen::placeholders::all;

namespace Optima {

struct OptimumStepper::Impl
{
    /// The options for the optimization calculation
    OptimumOptions options;

    /// The solution vector `s = [dx dy dz dw]`.
    Vector s;

    /// The right-hand side residual vector `r = [rx ry rz rw]`.
    Vector r;

    /// The `A` matrix in the saddle point equation assuming the ordering x = [x(free) x(fixed)].
    Matrix A;

    /// The `H` matrix in the saddle point equation assuming the ordering x = [x(free) x(fixed)].
    VariantMatrix H;

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

    /// The indices of the variables with lower bounds assuming the ordering x = [x(free) x(fixed)].
    VectorXi iwithlower;

    /// The indices of the variables with upper bounds assuming the ordering x = [x(free) x(fixed)].
    VectorXi iwithupper;

    /// The indices of the variables without lower bounds assuming the ordering x = [x(free) x(fixed)].
    VectorXi iwithoutlower;

    /// The indices of the variables without upper bounds assuming the ordering x = [x(free) x(fixed)].
    VectorXi iwithoutupper;

    /// Construct a OptimumStepper::Impl instance with given optimization problem structure.
    Impl(const OptimumStructure& structure)
    {
        // Initialize the members related to number of variables and constraints
        n  = structure.numVariables();
        m  = structure.numEqualityConstraints();
        nf = structure.variablesWithFixedValues().size();
        nx = n - nf;
        t  = 3*n + m;

        // The indices of the variables in the ordering x = [x(free) x(fixed)].
        const auto iordering = structure.orderingFixedValues();

        // Initialize the indices of the variables with lower bounds for the ordering x = [x(free) x(fixed)].
        iwithlower = iordering(structure.variablesWithLowerBounds());

        // Initialize the indices of the variables with upper bounds for the ordering x = [x(free) x(fixed)].
        iwithupper = iordering(structure.variablesWithLowerBounds());

        // Initialize the indices of the variables without lower bounds for the ordering x = [x(free) x(fixed)].
        iwithoutlower = iordering(structure.variablesWithoutLowerBounds());

        // Initialize the indices of the variables without upper bounds for the ordering x = [x(free) x(fixed)].
        iwithoutupper = iordering(structure.variablesWithoutLowerBounds());

        // Copy the matrix A with columns having order according to iordering
        A.noalias() = structure.equalityConstraintMatrix()(all, iordering);

        // Allocate memory for some members
        Z = zeros(n);
        W = zeros(n);
        L = zeros(n);
        U = zeros(n);
        r = zeros(t);
        s = zeros(t);

        // Initialize the saddle point solver
        solver.initialize(A);
    }

    /// Update the matrices Z, W, L, U that appear in the interior-point saddle point problem.
    auto updateZWLU(const OptimumParams& params, const OptimumState& state) -> void
    {
        // Initialize the diagonal matrix Z assuming the ordering x = [x(free) x(fixed)]
        Z(iwithlower) = state.z(iwithlower);
        Z(iwithoutlower).fill(0.0);
        Z.tail(nf).fill(0.0);

        // Initialize the diagonal matrix W assuming the ordering x = [x(free) x(fixed)]
        W(iwithupper) = state.w(iwithupper);
        W(iwithoutupper).fill(0.0);
        W.tail(nf).fill(0.0);

        // Initialize the diagonal matrix L assuming the ordering x = [x(free) x(fixed)]
        L(iwithlower) = state.x(iwithlower) - params.xlower;
        L(iwithoutlower).fill(1.0);
        L.tail(nf).fill(1.0);

        // Initialize the diagonal matrix U assuming the ordering x = [x(free) x(fixed)]
        U(iwithupper) = params.xupper - state.x(iwithupper);
        U(iwithoutupper).fill(1.0);
        U.tail(nf).fill(1.0);
    }

    /// Decompose the interior-point saddle point matrix.
    auto decompose(const OptimumParams& params, const OptimumState& state) -> Result
    {
        switch(state.H.structure()) {
        case MatrixStructure::Dense: return decomposeDenseHessianMatrix(params, state);
        case MatrixStructure::Diagonal: return decomposeDiagonalHessianMatrix(params, state);
        case MatrixStructure::Zero: return decomposeDiagonalHessianMatrix(params, state);
        }
        return {};
    }

    /// Decompose the interior-point saddle point matrix for diagonal Hessian matrices.
    auto decomposeDiagonalHessianMatrix(const OptimumParams& params, const OptimumState& state) -> Result
    {
        // The result of this method call
        Result res;

        // Update the matrices Z, W, L, U
        updateZWLU(params, state);

        // Set the structure of the Hessian matrix to diagonal
        H.setDiagonal(n);

        // Create a view for the block of H corresponding to free variables
        auto Hxx = H.diagonal().head(nx);

        // Copy values of the Hessian matrix to Hxx assuming the ordering x = [x(free) x(fixed)]
        Hxx.noalias() = state.H.diagonal().head(nx);

        // Define the interior-point saddle point matrix assuming the ordering x = [x(free) x(fixed)]
        IpSaddlePointMatrix spm(H, A, Z, W, L, U, nf);

        // Decompose the interior-point saddle point matrix
        solver.decompose(spm);

        return res.stop();
    }

    /// Decompose the interior-point saddle point matrix for dense Hessian matrices.
    auto decomposeDenseHessianMatrix(const OptimumParams& params, const OptimumState& state) -> Result
    {
        // The result of this method call
        Result res;

        // Update the matrices Z, W, L, U
        updateZWLU(params, state);

        // Set the structure of the Hessian matrix to dense
        H.setDense(n);

        // Create a view for the block of H corresponding to free variables
        auto Hxx = H.dense().topLeftCorner(nx, nx);

        // Copy values of the Hessian matrix to Hxx
        Hxx.noalias() = state.H.dense().topLeftCorner(nx, nx);

        // Define the interior-point saddle point matrix assuming the ordering x = [x(free) x(fixed)]
        IpSaddlePointMatrix spm(H, A, Z, W, L, U, nf);

        // Decompose the interior-point saddle point matrix
        solver.decompose(spm);

        return res.stop();
    }

    /// Solve the interior-point saddle point matrix.
    auto solve(const OptimumParams& params, const OptimumState& state) -> Result
    {
        // The result of this method call
        Result res;

        // Update the matrices Z, W, L, U
        updateZWLU(params, state);

        // Views to the sub-vectors in r = [a b c d]
        auto a = r.head(n);
        auto b = r.segment(n, m);
        auto c = r.segment(n + m, n);
        auto d = r.tail(n);

        // Views to the sub-vectors in a = [a(free) a(fixed)]
        auto ax = a.head(nx);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [c(free) c(fixed)]
        auto cx = c.head(nx);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [d(free) d(fixed)]
        auto dx = d.head(nx);
        auto df = d.tail(nf);

        // Views to the sub-vectors in z = [z(free) z(fixed)] and w = [w(free) w(fixed)]
        auto zx = state.z.head(nx);
        auto wx = state.w.head(nx);

        // Initialize the gradient of the objective function w.r.t. free and fixed variables
        const auto gx = state.g.head(nx);

        const auto x = state.x;
        const auto y = state.y;

        // Views to the sub-matrices in A = [Ax Af]
        const auto Ax = A.leftCols(nx);

        // Calculate ax = -(gx + tr(Ax)*y - zx + wx)
        ax.noalias() = -(gx + tr(Ax) * y - zx + wx);
        af.fill(0.0);

        // Calculate b = -(A*x - b)
        b.noalias() = -(A * x - params.b);

        // Calculate both c and d vectors
        cx.noalias() = options.mu - L % Z; // TODO Check if mu is still needed. Maybe this algorithm no longer needs perturbation.
        dx.noalias() = options.mu - U % W; // TODO Check if mu is still needed. Maybe this algorithm no longer needs perturbation.
        cf.fill(0.0);
        df.fill(0.0);

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
    auto matrix() const -> IpSaddlePointMatrix
    {
        return IpSaddlePointMatrix(H, A, Z, W, L, U, nf);
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

auto OptimumStepper::decompose(const OptimumParams& params, const OptimumState& state) -> Result
{
    return pimpl->decompose(params, state);
}

auto OptimumStepper::solve(const OptimumParams& params, const OptimumState& state) -> Result
{
    return pimpl->solve(params, state);
}

auto OptimumStepper::matrix() const -> IpSaddlePointMatrix
{
    return pimpl->matrix();
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

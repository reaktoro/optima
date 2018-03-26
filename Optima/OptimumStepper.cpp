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
using namespace Eigen;
using Eigen::placeholders::all;

namespace Optima {

struct OptimumStepper::Impl
{
    /// The structure of the optimization problem.
    OptimumStructure structure;

    /// The options for the optimization calculation
    OptimumOptions options;

    /// The solution vector `sol = [dx dy dz dw]`.
    Vector solution;

    /// The right-hand side residual vector `res = [rx ry rz rw]`.
    Vector residual;

    /// The `A` matrix in the saddle point equation.
    Matrix A;

    /// The `H` dense matrix in the saddle point equation.
    Matrix H;

    /// The `H` diagonal matrix in the saddle point equation.
    Vector Hdiag;

    /// The matrices Z, W, L, U
    Vector Z, W, L, U;

    /// The variables x arranged in the ordering x = [x(free) x(fixed)]
    Vector x;

    /// The gradient of the objective function arranged in the ordering g = [g(free) g(fixed)]
    Vector g;

    /// The ordering of the variables into [x(free) x(fixed)].
    VectorXi iordering;

    /// The number of variables.
    Index n;

    /// The current number of free and fixedvariables.
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

        // The ordering of the variables partitioned as [free variablesixed variables]
        iordering = structure.orderingFixedValues();

        // Copy the matrix A with columns having order accoring to iordering
        A.noalias() = structure.equalityConstraintMatrix()(all, iordering);

        // Allocate memory for some members
        H = zeros(n, n);
        g = zeros(n);
        x = zeros(n);
        Z = zeros(n);
        W = zeros(n);
        L = zeros(n);
        U = zeros(n);
        residual = zeros(t);
        solution = zeros(t);

        // Initialize the saddle point solver
        solver.initialize(A);
    }

    /// Update the matrices Z, W, L, U that appear in the interior-point saddle point problem.
    auto updateZWLU(const OptimumParams& params, const OptimumState& state) -> void
    {
        // The lower and upper bounds of x
        const auto xlower = params.lowerBounds();
        const auto xupper = params.upperBounds();

        // The indices of the variables with/without lower/upper bounds
        const auto iwithlower = structure.variablesWithLowerBounds();
        const auto iwithupper = structure.variablesWithUpperBounds();
        const auto iwithoutlower = structure.variablesWithoutLowerBounds();
        const auto iwithoutupper = structure.variablesWithoutUpperBounds();

        // Initialize the diagonal matrices Z and W
        Z(iwithlower) = state.z(iwithlower);
        W(iwithupper) = state.w(iwithupper);
        Z(iwithoutlower).fill(0.0);
        W(iwithoutupper).fill(0.0);

        // Initialize the diagonal matrices L and U
        L(iwithlower) = state.x(iwithlower) - xlower;
        U(iwithupper) = state.x(iwithupper) - xupper;
        L(iwithoutlower).fill(1.0);
        U(iwithoutupper).fill(1.0);

        // Permute the columns of A, Z, W, L and U according to ordering x = [x(free) x(fixed)]
//        Z = Z(iordering);
//        W = W(iordering);
//        L = L(iordering);
//        U = U(iordering);
//        VectorXd tmp;
//        Z = tmp.noalias() = Z(iordering);
//        W = tmp.noalias() = W(iordering);
//        L = tmp.noalias() = L(iordering);
//        U = tmp.noalias() = U(iordering);

        iordering.asPermutation().transpose().applyThisOnTheLeft(Z);
        iordering.asPermutation().transpose().applyThisOnTheLeft(W);
        iordering.asPermutation().transpose().applyThisOnTheLeft(L);
        iordering.asPermutation().transpose().applyThisOnTheLeft(U);
    }

    /// Decompose the interior-point saddle point matrix.
    auto decompose(const OptimumParams& params, const OptimumState& state) -> Result
    {
        switch(state.H.structure()) {
        case MatrixStructure::Dense: return decomposeDenseHessianMatrix(params, state);
        case MatrixStructure::Diagonal: return decomposeDiagonalHessianMatrix(params, state);
        case MatrixStructure::Zero: return decomposeDiagonalHessianMatrix(params, state);
        }
    }

    /// Decompose the interior-point saddle point matrix for diagonal Hessian matrices.
    auto decomposeDiagonalHessianMatrix(const OptimumParams& params, const OptimumState& state) -> Result
    {
        // The result of this method call
        Result res;

        // Update the matrices Z, W, L, U
        updateZWLU(params, state);

        // The indices of the free varibles
        const auto jx = iordering.head(nx);

        // Ensure suffient allocated memory for matrix Hdiag
        Hdiag.resize(n);

        // Create a view for the block of Hdiag corresponding to free variables
        auto Hxx = Hdiag.head(nx);

        // Copy values of the Hessian matrix to Hxx
        Hxx.noalias() = state.H.diagonal()(jx);

        // Define the interior-point saddle point matrix
        IpSaddlePointMatrix spm(Hdiag, A, Z, W, L, U, nf);

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

        // The indices of the free varibles
        const auto jx = iordering.head(nx);

        // Ensure suffient allocated memory for matrix H
        H.resize(n, n);

        // Create a view for the block of H corresponding to free variables
        auto Hxx = H.topLeftCorner(nx, nx);

        // Copy values of the Hessian matrix to Hxx
        Hxx.noalias() = state.H.dense()(jx, jx);

        // Define the interior-point saddle point matrix
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

        // The gradient of the objective function corresponding to free variables
        auto gx = g.head(nx);
        auto gf = g.tail(nf);

        // Views to the sub-vectors in residual = [a b c d]
        auto a = residual.head(n);
        auto b = residual.segment(n, m);
        auto c = residual.segment(n + m, n);
        auto d = residual.tail(n);

        // Views to the sub-vectors in a = [a(free) a(fixed)] = [ax af]
        auto ax = a.head(nx);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [c(free) c(fixed)] = [cx cf]
        auto cx = c.head(nx);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [d(free) d(fixed)] = [dx df]
        auto dx = d.head(nx);
        auto df = d.tail(nf);

        // Views to the sub-vectors in z = [z(free) z(fixed)] = [zx zf]
        auto zx = Z.head(nx);

        // Views to the sub-vectors in w = [w(free) w(fixed)] = [wx wf]
        auto wx = W.head(nx);

        // Views to the sub-vectors in the vector solution = [delta(x) delta(y) delta(z) delta(w)]
        auto hx = solution.head(n);
        auto hy = solution.segment(n, m);
        auto hz = solution.segment(n + m, n);
        auto hw = solution.tail(n);

        // The variables x arranged in the ordering x = [x(free) x(fixed)]
        x.noalias() = state.x(iordering);

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Initialize the gradient of the objective function w.r.t. free and fixed variables
        gx.noalias() = state.g(jx);
        gf.fill(0.0);

        // Views to the sub-matrices in A = [Ax Af]
        const auto Ax = A.leftCols(nx);

        // Calculate ax = -(gx + tr(Ax)*y - zx + wx)
        ax.noalias() = -(gx + tr(Ax) * state.y - zx + wx);
        af.fill(0.0);

        // Calculate b = -(A*x - b)
        b.noalias() = -(A*x - params.b());

        // Calculate both c and d vectors
        cx.noalias() = options.mu - L % Z;
        dx.noalias() = options.mu - U % W;
        cf.fill(0.0);
        df.fill(0.0);

        // Solve the saddle point problem
        solver.solve({a, b, c, d}, {hx, hy, hz, hw});

        // Permute the calculated (dx dz dw) to their original order
        iordering.asPermutation().applyThisOnTheLeft(hx);
        iordering.asPermutation().applyThisOnTheLeft(hz);
        iordering.asPermutation().applyThisOnTheLeft(hw);

        return res.stop();
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

auto OptimumStepper::step() const -> VectorConstRef
{
    return pimpl->solution;
}

auto OptimumStepper::dx() const -> VectorConstRef
{
    return pimpl->solution.head(pimpl->n);
}

auto OptimumStepper::dy() const -> VectorConstRef
{
    return pimpl->solution.segment(pimpl->n, pimpl->m);
}

auto OptimumStepper::dz() const -> VectorConstRef
{
    return pimpl->solution.tail(pimpl->n);
}

auto OptimumStepper::dw() const -> VectorConstRef
{
    return pimpl->solution.tail(pimpl->n);
}

auto OptimumStepper::residual() const -> VectorConstRef
{
    return pimpl->residual;
}

auto OptimumStepper::residualOptimality() const -> VectorConstRef
{
    return pimpl->residual.head(pimpl->n);
}

auto OptimumStepper::residualFeasibility() const -> VectorConstRef
{
    return pimpl->residual.segment(pimpl->n, pimpl->m);
}

auto OptimumStepper::residualComplementarityLowerBounds() const -> VectorConstRef
{
    return pimpl->residual.tail(pimpl->n);
}

auto OptimumStepper::residualComplementarityUpperBounds() const -> VectorConstRef
{
    return pimpl->residual.tail(pimpl->n);
}

} // namespace Optima

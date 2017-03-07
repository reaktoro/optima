// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

#include "SaddlePointSolver.hpp"

// Optima includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointOptions.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/EigenExtern.hpp>
using namespace Eigen;

namespace Optima {

struct SaddlePointSolver::Impl
{
    /// The canonicalizer of the Jacobian matrix *A*.
    Canonicalizer canonicalizer;

    /// The options used to solve the saddle point problems.
    SaddlePointOptions options;

    /// The number of rows and columns in the Jacobian matrix *A*
    Index m, n;

    /// The number of basic and non-basic.
    Index nb, nn;

    /// The number of free and fixed variables.
    Index nx, nf;

    /// The number of free and fixed basic variables.
    Index nbx, nbf;

    /// The number of free and fixed non-basic variables.
    Index nnx, nnf;

    /// The priority weights for the selection of basic variables.
    VectorXd weights;

    /// The matrix used as a workspace for the decompose and solve methods.
    MatrixXd mat;

    /// The vector used as a workspace for the decompose and solve methods.
    VectorXd vec;

    /// The ordering of the variables as (free-basic, free-non-basic, fixed-basic, fixed-non-basic)
    VectorXi iordering;

    /// The LU solver used to calculate *xb* when the Hessian matrix is in diagonal form.
    Eigen::PartialPivLU<MatrixXd> luxb;

    /// The LU solver used to calculate *xn* when the Hessian matrix is in dense form.
    Eigen::PartialPivLU<MatrixXd> luxn;

    /// The partial LU solver used to calculate both *x* and *y* simultaneously.
    Eigen::PartialPivLU<MatrixXd> luxy_partial;

    /// The full LU solver used to calculate both *x* and *y* simultaneously.
    Eigen::FullPivLU<MatrixXd> luxy_full;

    /// Canonicalize the coefficient matrix *A* of the saddle point problem.
    auto canonicalize(const MatrixXd& A) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Set the number of rows and columns in A
        m = A.rows();
        n = A.cols();

        // Allocate auxiliary memory
        mat.resize(n + m, n + m);
        vec.resize(n + m);
        iordering.resize(n);

        // Compute the canonical form of matrix A
        canonicalizer.compute(A);

        // Set the number of basic and non-basic variables
        nb = canonicalizer.numBasicVariables();
        nn = canonicalizer.numNonBasicVariables();

        /// Set the number of free and fixed variables
        nx = n;
        nf = 0;

        // Set the number of free and fixed basic variables
        nbx = nb;
        nbf = 0;

        // Set the number of free and fixed non-basic variables
        nnx = nn;
        nnf = 0;

        // Update the ordering of the variables
        iordering.head(nb) = canonicalizer.ibasic();
        iordering.tail(nn) = canonicalizer.inonbasic();

        return res.stop();
    }

    /// Update the canonical form of the coefficient matrix *A* of the saddle point problem.
    auto updateCanonicalForm(const SaddlePointMatrix& lhs) -> void
    {
        // Update the number of fixed and free variables
        nf = lhs.fixed().size();
        nx = n - nf;

        // Update the priority weights for the update of the canonical form
        weights.noalias() = abs(inv(lhs.H().diagonal()));

        // Set the priority weights of the fixed variables to decreasing negative values
        rows(weights, lhs.fixed()) = -linspace(nf, 1, nf);

        // Update the canonical form and the ordering of the variables
        canonicalizer.update(weights);

        // Check if rationalization of the canonical form should be performed
        if(options.rationalize)
            canonicalizer.rationalize(options.maxdenominator);

        // Get the updated indices of basic and non-basic variables
        auto ibasic = canonicalizer.ibasic();
        auto inonbasic = canonicalizer.inonbasic();

        // Find the number of fixed basic variables (those with weights below or equal to zero)
        nbf = 0; while(nbf < nb && weights[ibasic[nb - nbf - 1]] <= 0.0) ++nbf;

        // Find the number of fixed non-basic variables (those with weights below or equal to zero)
        nnf = 0; while(nnf < nn && weights[inonbasic[nn - nnf - 1]] <= 0.0) ++nnf;

        // Update the number of free basic and free non-basic variables
        nbx = nb - nbf;
        nnx = nn - nnf;

        // Update the ordering of the free variables
        iordering.head(nx).head(nbx) = canonicalizer.ibasic().head(nbx);
        iordering.head(nx).tail(nnx) = canonicalizer.inonbasic().head(nnx);

        // Update the ordering of the fixed variables
        iordering.tail(nf).head(nbf) = canonicalizer.ibasic().tail(nbf);
        iordering.tail(nf).tail(nnf) = canonicalizer.inonbasic().tail(nnf);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeLU(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();

        // The rows and columns of `S` corresponding to free basic and free non-basic variables
        auto Sx = S.topLeftCorner(nbx, nnx);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(2*nbx + nnx, 2*nbx + nnx);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(nbx, nbx).setIdentity(nbx, nbx);
        M.topRightCorner(nbx, nbx).setIdentity(nbx, nbx);

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.bottomRows(nbx).middleCols(nbx, nnx) = Sx;
        M.rightCols(nbx).middleRows(nbx, nnx)  = tr(Sx);

        // Set the zero block of M on the bottom-right corner
        M.bottomRightCorner(nbx, nbx).setZero();

        // Set the H block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = submatrix(lhs.H(), ivx, ivx);

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            luxy_partial.compute(M);
        else
            luxy_full.compute(M);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // The columns of matrix `S` corresponding to fixed non-basic variables
        auto Sbnf = S.rightCols(nnf);

        // The columns of identity matrix corresponding to fixed basic variables
        auto Ibf = identity(nb, nb).rightCols(nbf);

        // The rows of matrix `R` corresponding to free basic variables
        auto Rx = R.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        // The view in `vec` corresponding to values of `b` for linerly independent equation.
        auto bb = vec.segment(nx, nb);
        auto bx = bb.head(nbx);

        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to the right-hand side vector of the linear equation
        auto r = vec.head(nx + nbx);

        // Set the vectors `ax` and `af`
        ax.noalias() = rows(a, ivx);
        af.noalias() = rows(a, ivf);

        // Set the vector `bx`
        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;
//        bx.noalias() = Rx*b - Ibf*abf - Sbnf*anf;

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            r.noalias() = luxy_partial.solve(r);
        else
            r.noalias() = luxy_full.solve(r);

        // Compute the `y` vector without canonicalization
        y.noalias() = tr(Rx)*bx;

        // Permute back the variables x to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceDiagonal(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();

        // The rows and columns of `S` corresponding to free basic and free non-basic variables
        auto Sx = S.topLeftCorner(nbx, nnx);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Create auxiliary matrix views
        auto Hx  = mat.diagonal().head(nx);
        auto Hbx = Hx.head(nbx);
        auto Hnx = Hx.tail(nnx);
        auto B   = mat.bottomLeftCorner(m, n);
        auto T   = mat.topRightCorner(n, m);
        auto M   = mat.bottomRightCorner(m, m);
        auto Bx  = B.topLeftCorner(nbx, nnx);
        auto Tx  = T.topLeftCorner(nnx, nbx);
        auto Mx  = M.topLeftCorner(nbx, nbx);

        // Set `H` as the diagonal Hessian according to current canonical ordering
        Hx.noalias() = rows(lhs.H().diagonal(), ivx);

        // Compute the auxiliary matrices Bb and Bbn
        Bx.noalias() = Sx * diag(inv(Hnx));
        Tx.noalias() = tr(Sx) * diag(Hbx);

        // Compute the matrix Mbb for the xb equation
        Mx.noalias()  = Bx * Tx;
        Mx.noalias() += identity(nbx, nbx);

        // Compute the LU decomposition of Mx.
        luxb.compute(Mx);
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDiagonal(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // The columns of matrix `S` corresponding to free and fixed non-basic variables
        auto Sbnx = S.topLeftCorner(nbx, nnx);
        auto Sbnf = S.rightCols(nnf);

        // The columns of identity matrix corresponding to fixed basic variables
        auto Ibf = identity(nb, nb).rightCols(nbf);

        // The rows of matrix `R` corresponding to free basic variables
        auto Rx = R.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        // The view in `af` corresponding to free basic and free non-basic variables.
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);

        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to values of `b` for free basic variables.
        auto bb = vec.segment(nx, nb);
        auto bx = bb.head(nbx);

        // The view in `y` corresponding to values of `y` for free basic variables.
        auto yx = y.head(nbx);

        // Create auxiliary sub-matrix views
        auto Hx  = mat.diagonal().head(nx);
        auto B   = mat.bottomLeftCorner(m, n);
        auto T   = mat.topRightCorner(n, m);
        auto Hbx = Hx.head(nbx);
        auto Hnx = Hx.tail(nnx);
        auto Bx  = B.topLeftCorner(nbx, nnx);
        auto Tx  = T.topLeftCorner(nnx, nbx);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = rows(a, ivx);
        af.noalias() = rows(a, ivf);

        // Set the vector `bx`
        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;
//        bx.noalias() = Rx*b - Ibf*abf - Sbnf*anf;

        // Compute the saddle point problem solution
        yx.noalias()   = abx;
        anx.noalias() -= tr(Sbnx)*yx;
        bx.noalias()  -= Bx*anx;
        abx.noalias()  = luxb.solve(bx);
        anx.noalias() += Tx*abx;
        anx.noalias()  = diag(inv(Hnx))*anx;
        bx.noalias()   = yx;
        bx.noalias()  -= diag(Hbx)*abx;

        // Compute the y vector without canonicalization
        y.noalias() = tr(Rx)*bx;

        // Permute back the variables `x` to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspace(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();

        // Create auxiliary matrix views
        auto Hx   = mat.topLeftCorner(nx, nx);
        auto B    = mat.bottomLeftCorner(m, n);
        auto Hbbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbnx = Hx.topRightCorner(nbx, nnx);
        auto Hnbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnnx = Hx.bottomRightCorner(nnx, nnx);
        auto Bbnx = B.topLeftCorner(nbx, nnx);
        auto Sbnx = S.topLeftCorner(nbx, nnx);
        auto Mnnx = Hnnx;

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Set `H` as the diagonal Hessian according to current canonical ordering
        Hx.noalias() = submatrix(lhs.H(), ivx, ivx);

        // Calculate auxiliary matrix Bbn
        Bbnx.noalias()  = Hbbx * Sbnx;
        Bbnx.noalias() -= Hbnx;

        // Calculate coefficient matrix Mnn for the xn equation.
        Mnnx.noalias() += tr(Sbnx) * Bbnx;
        Mnnx.noalias() -= Hnbx * Sbnx;

        // Compute the LU decomposition of Mnnx.
        luxn.compute(Mnnx);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspace(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // The columns of matrix `S` corresponding to free and fixed non-basic variables
        auto Sbnx = S.topLeftCorner(nbx, nnx);
        auto Sbnf = S.rightCols(nnf);

        // The columns of identity matrix corresponding to fixed basic variables
        auto Ibf = identity(nb, nb).rightCols(nbf);

        // The rows of matrix `R` corresponding to free basic variables
        auto Rx = R.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        // The view in `af` corresponding to free basic and free non-basic variables.
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);

        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to values of `b` for free basic variables.
        auto bb = vec.segment(nx, nb);
        auto bx = bb.head(nbx);

        // The view in `y` corresponding to values of `y` for free basic variables.
        auto yx = y.head(nbx);

        // Create auxiliary matrix views
        auto Hx   = mat.topLeftCorner(nx, nx);
        auto Hbbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbnx = Hx.topRightCorner(nbx, nnx);
        auto Hnbx = Hx.bottomLeftCorner(nnx, nbx);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = rows(a, ivx);
        af.noalias() = rows(a, ivf);

        // Set the vector `bx`
        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;
//        bx.noalias() = Rx*b - Ibf*abf - Sbnf*anf;

        auto xx = x.head(nx);
        auto xbx = xx.head(nbx);
        auto xnx = xx.tail(nnx);

        // Compute the saddle point problem solution
        yx.noalias()  = abx - Hbbx*bx;
        xnx.noalias() = anx - Hnbx*bx - tr(Sbnx)*yx;
        anx.noalias() = luxn.solve(xnx);
        xbx.noalias() = abx;
        abx.noalias() = bx - Sbnx*anx;
        bx.noalias()  = xbx - Hbbx*abx - Hbnx*anx;

        // Compute the y vector without canonicalization
        y.noalias() = tr(Rx)*bx;

        // Permute back the variables `x` to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Perform the decomposition according to the chosen method
        switch(options.method)
        {
        case SaddlePointMethod::PartialPivLU:
        case SaddlePointMethod::FullPivLU:
            decomposeLU(lhs); break;
        case SaddlePointMethod::RangespaceDiagonal:
            decomposeRangespaceDiagonal(lhs); break;
        default:
            decomposeNullspace(lhs); break;
        }

        return res.stop();
    }

    /// Solve the saddle point problem with diagonal Hessian matrix.
    auto solve(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Perform the solution according to the chosen method
        switch(options.method)
        {
        case SaddlePointMethod::PartialPivLU:
        case SaddlePointMethod::FullPivLU:
            solveLU(rhs, sol); break;
        case SaddlePointMethod::RangespaceDiagonal:
            solveRangespaceDiagonal(rhs, sol); break;
        default:
            solveNullspace(rhs, sol); break;
        }

        return res.stop();
    }

    auto solve(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Assemble the saddle point coefficient matrix and the right-hand side vector
        mat << lhs;
        vec << rhs;

        // Solve the saddle point problem
        vec.noalias() = mat.lu().solve(vec);

        // Set the saddle point solution with values in vec
        sol = vec;

        return res.stop();
    }
};

SaddlePointSolver::SaddlePointSolver()
: pimpl(new Impl())
{}

SaddlePointSolver::SaddlePointSolver(const SaddlePointSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolver::~SaddlePointSolver()
{}

auto SaddlePointSolver::operator=(SaddlePointSolver other) -> SaddlePointSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolver::setOptions(const SaddlePointOptions& options) -> void
{
    pimpl->options = options;
}

auto SaddlePointSolver::setMethodMoreEfficient(Index n, Index m) -> void
{
}

auto SaddlePointSolver::setMethodMoreAccurate(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs) -> void
{
}

auto SaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->options;
}

auto SaddlePointSolver::canonicalize(const MatrixXd& A) -> SaddlePointResult
{
    return pimpl->canonicalize(A);
}

auto SaddlePointSolver::decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
{
    return pimpl->decompose(lhs);
}

auto SaddlePointSolver::solve(SaddlePointVector rhs, SaddlePointSolution sol) -> SaddlePointResult
{
    return pimpl->solve(rhs, sol);
}

auto SaddlePointSolver::solve(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution sol) -> SaddlePointResult
{
    return pimpl->solve(lhs, rhs, sol);
}

} // namespace Optima

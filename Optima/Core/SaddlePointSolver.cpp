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
#include <Optima/Core/HessianMatrix.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/EigenExtern.hpp>
using namespace Eigen;

namespace Optima {

struct SaddlePointSolver::Impl
{
    /// The canonicalizer of the Jacobian matrix *A*.
    Canonicalizer canonicalizer;

    /// The method used to solve the saddle point problems
    SaddlePointMethod method = SaddlePointMethod::PartialPivLU;

    /// The number of rows and columns in the Jacobian matrix *A*
    Index m, n;

    /// The number of basic, non-basic, and fixed variables.
    Index nb, nn, nf;

    /// The priority weights for the selection of basic variables.
    VectorXd w;

    /// The matrix used as a workspace for the decompose and solve methods.
    MatrixXd mat;

    /// The vector used as a workspace for the decompose and solve methods.
    VectorXd vec;

    /// The LU solver used to calculate *xb* when the Hessian matrix is in diagonal form.
    Eigen::PartialPivLU<MatrixXd> luxb;

    /// The LU solver used to calculate *xn* when the Hessian matrix is in dense form.
    Eigen::PartialPivLU<MatrixXd> luxn;

    /// The partial LU solver used to calculate both *x* and *y* simultaneously.
    Eigen::PartialPivLU<MatrixXd> luxy_partial;

    /// The full LU solver used to calculate both *x* and *y* simultaneously.
    Eigen::FullPivLU<MatrixXd> luxy_full;

    /// The mode of the Hessian matrix decomposed last time
    HessianMatrix::Mode hessian_mode;

    /// Canonicalize the coefficient matrix *A* of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Get the Jacobian matrix
        auto A = lhs.A();

        // Set the number of rows and columns in A
        m = A.rows();
        n = A.cols();

        // Allocate auxiliary memory
        mat.resize(n + m, n + m);
        vec.resize(n + m);

        // Compute the canonical form of matrix A
        canonicalizer.compute(A);

        return res.stop();
    }

    /// Update the canonical form of the coefficient matrix *A* of the saddle point problem.
    auto updateCanonicalForm(const SaddlePointMatrix& lhs) -> void
    {
        // Get the Hessian matrix
        auto H = lhs.H();

        // Get the indices of fixed variables
        const Indices& fixed = lhs.fixed();

        // Update the priority weights for the update of the canonical form
        if(H.isdiagonal()) w.noalias() = inv(H.diagonal());
                      else w.noalias() = inv(H.dense().diagonal());

        // Update the canonical form and the ordering of the variables
        canonicalizer.update(w, fixed);

        // Update the number of fixed, basic, and non-basic variables
        nf = fixed.size();
        nb = canonicalizer.rows();
        nn = n - nb - nf;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeLU(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();

        // Create a view to the non-basic columns of S (ignoring the fixed variables)
        auto Sbn = S.leftCols(nn);

        // Create a view to the basic and non-basic rows of Q (ignoring the fixed variables)
        auto Qbn = Q.head(nb + nn);

        // Create a view to the M block of the auxiliary matrix mat where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(2*nb + nn, 2*nb + nn);

        // Create a view to the H block of the canonical saddle point matrix
        auto H = M.topLeftCorner(nb + nn, nb + nn);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(nb, nb).setIdentity(nb, nb);
        M.topRightCorner(nb, nb).setIdentity(nb, nb);

        // Set the Sbn and tr(Sbn) blocks in the canonical saddle point matrix
        M.bottomRows(nb).middleCols(nb, nn) = Sbn;
        M.rightCols(nb).middleRows(nb, nn)  = tr(Sbn);

        // Set the H block of the canonical saddle point matrix
        if(lhs.H().isdense()) H.noalias() = submatrix(lhs.H().dense(), Qbn, Qbn);
        else H = diag(rows(lhs.H().diagonal(), Qbn));

        // Compute the LU decomposition of M.
        if(method == SaddlePointMethod::PartialPivLU) luxy_partial.compute(M);
        else luxy_full.compute(M);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // Create auxiliary sub-matrix views
        auto H   = mat.topLeftCorner(n, n).diagonal();
        auto B   = mat.bottomLeftCorner(m, n);
        auto T   = mat.topRightCorner(n, m);
        auto Hb  = H.head(nb);
        auto Hn  = H.segment(nb, nn);
        auto Bbn = B.topLeftCorner(nb, nn);
        auto Tnb = T.topLeftCorner(nn, nb);
        auto Sbn = S.leftCols(nn);
        auto Sbf = S.rightCols(nf);

        // Create auxiliary sub-vector views
        auto a  = vec.head(n);
        auto b  = vec.tail(m);
        auto ab = a.head(nb);
        auto an = a.segment(nb, nn);
        auto af = a.tail(nf);
        auto bb = b.head(nb);
        auto yb = y.head(nb);

        // Reorder vector a in the canonical order
        a.noalias() = rows(rhs.a(), Q);

        // Apply the regularizer matrix to b
        bb.noalias() = R*rhs.b();

        // Compute the saddle point problem solution
        yb.noalias()  = ab;
        an.noalias() -= tr(Sbn)*yb;
        bb.noalias() -= Sbf*af;
        bb.noalias() -= Bbn*an;
        ab.noalias()  = luxb.solve(bb);
        an.noalias() += Tnb*ab;
        an.noalias()  = diag(inv(Hn))*an;
        bb.noalias()  = yb;
        bb.noalias() -= diag(Hb)*ab;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R)*bb;

        // Permute back the variables x to their original ordering
        rows(x, Q).noalias() = a;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceDiagonal(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();

        // Create auxiliary matrix views
        auto H  = mat.topLeftCorner(n, n).diagonal();
        auto B  = mat.bottomLeftCorner(m, n);
        auto T  = mat.topRightCorner(n, m);
        auto M  = mat.bottomRightCorner(m, m);
        auto Hb = H.head(nb);
        auto Hn = H.segment(nb, nn);
        auto Bbn = B.topLeftCorner(nb, nn);
        auto Tnb = T.topLeftCorner(nn, nb);
        auto Mbb = M.topLeftCorner(nb, nb);
        auto Sbn = S.leftCols(nn);

        // Set `H` as the diagonal Hessian according to current canonical ordering
        H.noalias() = rows(lhs.H().diagonal(), Q);

        // Compute the auxiliary matrices Bb and Bbn
        Bbn.noalias() = Sbn * diag(inv(Hn));
        Tnb.noalias() = tr(Sbn) * diag(Hb);

        // Compute the matrix Mbb for the xb equation
        Mbb.noalias()  = Bbn * Tnb;
        Mbb.noalias() += identity(nb, nb);

        // Compute the LU decomposition of Mbb.
        luxb.compute(Mbb);
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDiagonal(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // Create auxiliary sub-matrix views
        auto H   = mat.topLeftCorner(n, n).diagonal();
        auto B   = mat.bottomLeftCorner(m, n);
        auto T   = mat.topRightCorner(n, m);
        auto Hb  = H.head(nb);
        auto Hn  = H.segment(nb, nn);
        auto Bbn = B.topLeftCorner(nb, nn);
        auto Tnb = T.topLeftCorner(nn, nb);
        auto Sbn = S.leftCols(nn);
        auto Sbf = S.rightCols(nf);

        // Create auxiliary sub-vector views
        auto a  = vec.head(n);
        auto b  = vec.tail(m);
        auto ab = a.head(nb);
        auto an = a.segment(nb, nn);
        auto af = a.tail(nf);
        auto bb = b.head(nb);
        auto yb = y.head(nb);

        // Reorder vector a in the canonical order
        a.noalias() = rows(rhs.a(), Q);

        // Apply the regularizer matrix to b
        bb.noalias() = R*rhs.b();

        // Compute the saddle point problem solution
        yb.noalias()  = ab;
        an.noalias() -= tr(Sbn)*yb;
        bb.noalias() -= Sbf*af;
        bb.noalias() -= Bbn*an;
        ab.noalias()  = luxb.solve(bb);
        an.noalias() += Tnb*ab;
        an.noalias()  = diag(inv(Hn))*an;
        bb.noalias()  = yb;
        bb.noalias() -= diag(Hb)*ab;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R)*bb;

        // Permute back the variables x to their original ordering
        rows(x, Q).noalias() = a;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspace(const SaddlePointMatrix& lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();

        // Create auxiliary matrix views
        auto H  = mat.topLeftCorner(n, n);
        auto B  = mat.bottomLeftCorner(m, n);
        auto Htl = H.topLeftCorner(nb + nn, nb + nn);
        auto Hbb = Htl.topLeftCorner(nb, nb);
        auto Hbn = Htl.topRightCorner(nb, nn);
        auto Hnb = Htl.bottomLeftCorner(nn, nb);
        auto Hnn = Htl.bottomRightCorner(nn, nn);
        auto Bbn = B.topLeftCorner(nb, nn);
        auto Mnn = Hnn;
        auto Sbn = S.leftCols(nn);

        // Set `H` as the diagonal Hessian according to current canonical ordering
        H.noalias() = submatrix(lhs.H().dense(), Q, Q);

        // Calculate auxiliary matrix Bbn
        Bbn.noalias()  = Hbb * Sbn;
        Bbn.noalias() -= Hbn;

        // Calculate coefficient matrix Mnn for the xn equation.
        Mnn.noalias() += tr(Sbn) * Bbn;
        Mnn.noalias() -= Hnb * Sbn;

        // Compute the LU decomposition of Mnn.
        luxn.compute(Mnn);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspace(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q().indices();
        const auto& S = canonicalizer.S();
        const auto& R = canonicalizer.R();

        // Create auxiliary matrix views
        auto H   = mat.topLeftCorner(n, n);
        auto Htl = H.topLeftCorner(nb + nn, nb + nn);
        auto Hbb = Htl.topLeftCorner(nb, nb);
        auto Hbn = Htl.topRightCorner(nb, nn);
        auto Hnb = Htl.bottomLeftCorner(nn, nb);
        auto Sbn = S.leftCols(nn);
        auto Sbf = S.rightCols(nf);

        // Create auxiliary sub-vector views
        auto a  = vec.head(n);
        auto b  = vec.tail(m);
        auto ab = a.head(nb);
        auto an = a.segment(nb, nn);
        auto af = a.tail(nf);
        auto bb = b.head(nb);
        auto xb = x.head(nb);
        auto xn = x.segment(nb, nn);
        auto yb = y.head(nb);

        // Reorder vector a in the canonical order
        a.noalias() = rows(rhs.a(), Q);

        // Apply the regularizer matrix to b
        bb.noalias() = R*rhs.b() - Sbf*af;

        // Compute the saddle point problem solution
        yb.noalias()  = ab - Hbb*bb;
        xn.noalias()  = an - Hnb*bb - tr(Sbn)*yb;
        an.noalias()  = luxn.solve(xn);
        xb.noalias()  = ab;
        ab.noalias()  = bb - Sbn*an;
        bb.noalias()  = xb - Hbb*ab - Hbn*an;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R)*bb;

        // Permute back the variables x to their original ordering
        rows(x, Q).noalias() = a;
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Set the mode of the Hessian matrix
        hessian_mode = lhs.H().mode();

        // Perform the decomposition according to the mode of the Hessian matrix
        switch(hessian_mode) {
        case HessianMatrix::Diagonal: decomposeRangespaceDiagonal(lhs); break;
                             default: decomposeNullspace(lhs); break; }

        return res.stop();
    }

    /// Solve the saddle point problem with diagonal Hessian matrix.
    auto solve(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Perform the solution according to the mode of the Hessian matrix
        switch(method) {
        case SaddlePointMethod::PartialPivLU:
        case SaddlePointMethod::RangespaceDiagonal:
            solveRangespaceDiagonal(rhs, sol); break;
                             default: solveNullspace(rhs, sol); break; }

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

auto SaddlePointSolver::setMethod(SaddlePointMethod method) -> void
{
    pimpl->method = method;
}

auto SaddlePointSolver::setMethodPartialPivLU() -> void
{
    setMethod(SaddlePointMethod::PartialPivLU);
}

auto SaddlePointSolver::setMethodFullPivLU() -> void
{
    setMethod(SaddlePointMethod::FullPivLU);
}

auto SaddlePointSolver::setMethodRangespaceDiagonal() -> void
{
    setMethod(SaddlePointMethod::RangespaceDiagonal);
}

auto SaddlePointSolver::setMethodNullspace() -> void
{
    setMethod(SaddlePointMethod::Nullspace);
}

auto SaddlePointSolver::setMethodMoreEfficient(Index n, Index m) -> void
{
}

auto SaddlePointSolver::setMethodMoreAccurate(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs) -> void
{
}

auto SaddlePointSolver::method() const -> SaddlePointMethod
{
    return pimpl->method;
}

auto SaddlePointSolver::canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
{
    return pimpl->canonicalize(lhs);
}

auto SaddlePointSolver::decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
{
    return pimpl->decompose(lhs);
}

auto SaddlePointSolver::solve(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult
{
    return pimpl->solve(rhs, sol);
}

auto SaddlePointSolver::solve(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult
{
    return pimpl->solve(lhs, rhs, sol);
}

} // namespace Optima

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



#include <iostream>



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

    /// The number of pivot free basic variables.
    Index nbx1, nnx1;

    /// The number of non-pivot free non-basic variables.
    Index nbx2, nnx2;

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
    auto canonicalize(MatrixXdConstRef A) -> SaddlePointResult
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

        // Set the number of pivot free basic variables.
        nbx1 = nbx;
        nnx1 = nnx;

        // Set the number of non-pivot free non-basic variables.
        nbx2 = 0;
        nnx2 = 0;

        // Initialize the ordering of the variables
        iordering.head(nb) = canonicalizer.ibasic();
        iordering.tail(nn) = canonicalizer.inonbasic();

        return res.stop();
    }

    /// Update the canonical form of the coefficient matrix *A* of the saddle point problem.
    auto updateCanonicalForm(SaddlePointMatrix lhs) -> void
    {
        // Update the number of fixed and free variables
        nf = lhs.fixed().size();
        nx = n - nf;

        // The diagonal entries of the Hessian matrix
        auto D = lhs.H().diagonal();

        // Update the priority weights for the update of the canonical form
        weights.noalias() = abs(inv(D));

        // Set the priority weights of the fixed variables to decreasing negative values
        rows(weights, lhs.fixed()) = -linspace(nf, 1, nf);

        // Update the canonical form and the ordering of the variables
        canonicalizer.update(weights);

        // Check if rationalization of the canonical form should be performed
        if(options.rationalize)
            canonicalizer.rationalize(options.maxdenominator);

        // Get the updated indices of basic and non-basic variables
        const auto& ibasic = canonicalizer.ibasic();
        const auto& inonbasic = canonicalizer.inonbasic();

        // Get the S matrix of the canonical form of A
        auto S = canonicalizer.S();

        // Find the number of fixed basic variables (those with weights below or equal to zero)
        nbf = 0; while(nbf < nb && weights[ibasic[nb - nbf - 1]] <= 0.0) ++nbf;

        // Find the number of fixed non-basic variables (those with weights below or equal to zero)
        nnf = 0; while(nnf < nn && weights[inonbasic[nn - nnf - 1]] <= 0.0) ++nnf;

        // Update the number of free basic and free non-basic variables
        nbx = nb - nbf;
        nnx = nn - nnf;

        // Update the number of non-pivot free basic variables.
        nbx2 = 0; while(nbx2 < nbx && weights[ibasic[nbx2]] > 1.0) ++nbx2;

        // Update the number of non-pivot free non-basic variables.
        nnx2 = 0; while(nnx2 < nnx && weights[inonbasic[nnx2]] > norminf(S.col(nnx2))) ++nnx2;

        // Update the number of pivot free basic and non-basic variables.
        nbx1 = nbx - nbx2;
        nnx1 = nnx - nnx2;

        // Update the ordering of the free variables
        iordering.head(nx).head(nbx) = canonicalizer.ibasic().head(nbx);
        iordering.head(nx).tail(nnx) = canonicalizer.inonbasic().head(nnx);

        // Update the ordering of the fixed variables
        iordering.tail(nf).head(nbf) = canonicalizer.ibasic().tail(nbf);
        iordering.tail(nf).tail(nnf) = canonicalizer.inonbasic().tail(nnf);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeLU_ZeroG(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sbnx = canonicalizer.S().topLeftCorner(nbx, nnx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(2*nbx + nnx, 2*nbx + nnx);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(nbx, nbx).setIdentity(nbx, nbx);
        M.topRightCorner(nbx, nbx).setIdentity(nbx, nbx);

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.bottomRows(nbx).middleCols(nbx, nnx) = Sbnx;
        M.rightCols(nbx).middleRows(nbx, nnx)  = tr(Sbnx);

        // Set the G block of M on the bottom-right corner
        M.bottomRightCorner(nbx, nbx).setZero();

        // Set the H block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = submatrix(lhs.H(), ivx, ivx);

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            luxy_partial.compute(M);
        else
            luxy_full.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeLU_DenseG(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sx = canonicalizer.S().leftCols(nnx);
        auto R = canonicalizer.R();

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(m + nx, m + nx);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(m, m).setIdentity(m, m);
        M.topRightCorner(m, m).setIdentity(m, m);

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.bottomRows(m).middleCols(m, nnx) = Sx;
        M.rightCols(m).middleRows(m, nnx)  = tr(Sx);

        // Set the G block of M on the bottom-right corner
        M.bottomRightCorner(m, m) = R * lhs.G() * tr(R);

        // Set the H block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = submatrix(lhs.H(), ivx, ivx);

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            luxy_partial.compute(M);
        else
            luxy_full.compute(M);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU_ZeroG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R().topRows(nbx);

        // The columns of matrix `S` corresponding to fixed non-basic variables
//        auto Sbnf = S.rightCols(nnf);
        auto Sbnf = S.topRightCorner(nbx, nnf);

        // The columns of identity matrix corresponding to fixed basic variables
//        auto Ibf = identity(nb, nb).rightCols(nbf);

        // The rows of matrix `R` corresponding to free basic variables
//        auto Rx = R.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        // The view in `vec` corresponding to values of `b` for linerly independent equation.
        auto bb = vec.segment(nx, nbx);
//        auto bx = bb.head(nbx);

        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to the right-hand side vector of the linear equation
        auto r = vec.head(nx + nbx);
//        std::cout << "r1 = " << tr(r) << std::endl;
//        r.fill(0.0);
//        std::cout << "r2 = " << tr(r) << std::endl;

        // Set the vectors `ax` and `af`
        ax.noalias() = rows(a, ivx);
//        std::cout << "r3 = " << tr(r) << std::endl;
        af.noalias() = rows(a, ivf);
//        std::cout << "r4 = " << tr(r) << std::endl;

        // Set the vector `bx`
//        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;
        bb.noalias() = R*b - Sbnf*anf;
//        std::cout << "r5 = " << tr(r) << std::endl;

//        std::cout << "bb = " << tr(bb) << std::endl;

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            r.noalias() = luxy_partial.solve(r);
        else
            r.noalias() = luxy_full.solve(r);


        std::cout << "r = " << tr(r) << std::endl;


        // Compute the `y` vector without canonicalization
//        y.noalias() = tr(Rx)*bx;
        y.noalias() = tr(R)*bb;

        // Permute back the variables x to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU_DenseG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // The columns of matrix `S` corresponding to fixed non-basic variables
//        auto Sbnf = S.rightCols(nnf);
        auto Sbnf = S.rightCols(nnf);

        // The columns of identity matrix corresponding to fixed basic variables
        auto Ibf = identity(m, m).leftCols(nb).rightCols(nbf);

        // The rows of matrix `R` corresponding to free basic variables
//        auto Rx = R.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        // The view in `vec` corresponding to values of `b` for linerly independent equation.
        auto bb = vec.segment(nx, m);
//        auto bx = bb.head(nbx);

        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to the right-hand side vector of the linear equation
        auto r = vec.head(nx + m);
//        std::cout << "r1 = " << tr(r) << std::endl;
//        r.fill(0.0);
//        std::cout << "r2 = " << tr(r) << std::endl;

        // Set the vectors `ax` and `af`
        ax.noalias() = rows(a, ivx);
//        std::cout << "r3 = " << tr(r) << std::endl;
        af.noalias() = rows(a, ivf);
//        std::cout << "r4 = " << tr(r) << std::endl;

        // Set the vector `bx`
        bb.noalias() = R * rhs.b() - Ibf*abf - Sbnf*anf;
//        bb.noalias() = R*b - Sbnf*anf;
//        std::cout << "r5 = " << tr(r) << std::endl;

//        std::cout << "bb = " << tr(bb) << std::endl;

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            r.noalias() = luxy_partial.solve(r);
        else
            r.noalias() = luxy_full.solve(r);


        std::cout << "r = " << tr(r) << std::endl;


        // Compute the `y` vector without canonicalization
//        y.noalias() = tr(Rx)*bx;
        y.noalias() = tr(R)*bb;

        // Permute back the variables x to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

//    /// Solve the saddle point problem using a LU decomposition method.
//    auto solveLU(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
//    {
//        // Alias to members of the saddle point vector solution.
//        auto x = sol.x();
//        auto y = sol.y();
//
//        // Alias to members of the saddle point right-hand side vector.
//        auto a = rhs.a();
//        auto b = rhs.b();
//
//        // Alias to the matrices of the canonicalization process
//        auto S = canonicalizer.S();
//        auto R = canonicalizer.R().topRows(nbx);
//
//        // The columns of matrix `S` corresponding to fixed non-basic variables
////        auto Sbnf = S.rightCols(nnf);
//        auto Sbnf = S.topRightCorner(nbx, nnf);
//
//        // The columns of identity matrix corresponding to fixed basic variables
////        auto Ibf = identity(nb, nb).rightCols(nbf);
//
//        // The rows of matrix `R` corresponding to free basic variables
////        auto Rx = R.topRows(nbx);
//
//        // The indices of the free and fixed variables
//        auto ivx = iordering.head(nx);
//        auto ivf = iordering.tail(nf);
//
//        // The view in `vec` corresponding to values of `a` for free and fixed variables.
//        auto ax = vec.head(nx);
//        auto af = vec.tail(nf);
//
//        // The view in `vec` corresponding to values of `b` for linerly independent equation.
//        auto bb = vec.segment(nx, nbx);
////        auto bx = bb.head(nbx);
//
//        // The view in `af` corresponding to fixed basic and fixed non-basic variables.
//        auto abf = af.head(nbf);
//        auto anf = af.tail(nnf);
//
//        // The view in `vec` corresponding to the right-hand side vector of the linear equation
//        auto r = vec.head(nx + nbx);
//        std::cout << "r1 = " << tr(r) << std::endl;
//        r.fill(0.0);
//        std::cout << "r2 = " << tr(r) << std::endl;
//
//        // Set the vectors `ax` and `af`
//        ax.noalias() = rows(a, ivx);
//        std::cout << "r3 = " << tr(r) << std::endl;
//        af.noalias() = rows(a, ivf);
//        std::cout << "r4 = " << tr(r) << std::endl;
//
//        // Set the vector `bx`
////        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;
//        bb.noalias() = R*b - Sbnf*anf;
//        std::cout << "r5 = " << tr(r) << std::endl;
//
//        std::cout << "bb = " << tr(bb) << std::endl;
//
//        // Compute the LU decomposition of M.
//        if(options.method == SaddlePointMethod::PartialPivLU)
//            r.noalias() = luxy_partial.solve(r);
//        else
//            r.noalias() = luxy_full.solve(r);
//
//
//        std::cout << "r = " << tr(r) << std::endl;
//
//
//        // Compute the `y` vector without canonicalization
////        y.noalias() = tr(Rx)*bx;
//        y.noalias() = tr(R)*bb;
//
//        // Permute back the variables x to their original ordering
//        rows(x, ivx).noalias() = ax;
//        rows(x, ivf).noalias() = af;
//    }

    auto decomposeLU(SaddlePointMatrix lhs) -> void
    {
        if(lhs.G().size()) decomposeLU_DenseG(lhs);
        else decomposeLU_ZeroG(lhs);
    }

    auto solveLU(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        if(lhs.G().size()) solveLU_DenseG(lhs, rhs, sol);
        else solveLU_ZeroG(lhs, rhs, sol);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceDiagonal(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        std::cout << "H = " << tr(lhs.H().diagonal()) << std::endl;

        auto ivx = iordering.head(nx);
        auto ivb = ivx.head(nbx);
        auto ivn = ivx.tail(nnx);

        auto Sbnx  = canonicalizer.S().topLeftCorner(nbx, nnx);
        auto Sb1n2 = Sbnx.bottomLeftCorner(nbx1, nnx2);
        auto Sb2n2 = Sbnx.topLeftCorner(nbx2, nnx2);
        auto Sbn1  = Sbnx.rightCols(nnx1);

        auto R = canonicalizer.R().topRows(nbx);

        auto H     = mat.col(0).head(nx);
        auto Hbb   = H.head(nbx);
        auto Hnn   = H.tail(nnx);
        auto Hb1b1 = Hbb.tail(nbx1);
        auto Hb2b2 = Hbb.head(nbx2);
        auto Hn1n1 = Hnn.tail(nnx1);
        auto Hn2n2 = Hnn.head(nnx2);

        auto G     = mat.bottomLeftCorner(nbx, nbx);
        auto Gb2b2 = G.topLeftCorner(nbx2, nbx2);
        auto Gb2b1 = G.topRightCorner(nbx2, nbx1);
        auto Gb1b2 = G.bottomLeftCorner(nbx1, nbx2);
        auto Gb1b1 = G.bottomRightCorner(nbx1, nbx1);

        auto T = mat.topRightCorner(nbx, nbx);
        auto Tb2b2 = T.topLeftCorner(nbx2, nbx2);
        auto Tb2b1 = T.topRightCorner(nbx2, nbx1);
        auto Tb1b2 = T.bottomLeftCorner(nbx1, nbx2);
        auto Tb1b1 = T.bottomRightCorner(nbx1, nbx1);

        auto B = mat.rightCols(nbx).middleRows(nbx, nnx1);

        auto M   = mat.bottomRightCorner(nbx + nnx2, nbx + nnx2);
        auto Mt  = M.topRows(nbx2);
        auto Mtl = Mt.leftCols(nbx2);
        auto Mtm = Mt.middleCols(nbx2, nbx1);
        auto Mtr = Mt.rightCols(nnx2);
        auto Mm  = M.middleRows(nbx2, nbx1);
        auto Mml = Mm.leftCols(nbx2);
        auto Mmm = Mm.middleCols(nbx2, nbx1);
        auto Mmr = Mm.rightCols(nnx2);
        auto Mb  = M.bottomRows(nnx2);
        auto Mbl = Mb.leftCols(nbx2);
        auto Mbm = Mb.middleCols(nbx2, nbx1);
        auto Mbr = Mb.rightCols(nnx2);






        mat.fill(0.0);






        // Clears the block where Hn2n2 (diagonal) is assigned to avoid dirty off-diagonal entries
        Mbr.fill(0.0);

        std::cout << "mat = 1\n" << mat << std::endl;
//        Hbb.noalias() = rows(lhs.H().diagonal(), ivb);
        H.noalias() = rows(lhs.H().diagonal(), ivx);
        std::cout << "mat = 2\n" << mat << std::endl;
//        Hnn.noalias() = rows(lhs.H().diagonal(), ivn);
        std::cout << "mat = 3\n" << mat << std::endl;
        G.noalias() = R * lhs.G() * tr(R);
        std::cout << "mat = 4\n" << mat << std::endl;
        B.noalias() = diag(inv(Hn1n1)) * tr(Sbn1);
        std::cout << "mat = 5\n" << mat << std::endl;
        T.noalias() = Sbn1 * B;
        std::cout << "mat = 6\n" << mat << std::endl;

        if(lhs.G().size())
        {
            Mtl.noalias()   = (Tb2b2 - Gb2b2)*diag(Hb2b2);
            Mtl.diagonal() += ones(nbx2);
            std::cout << "mat = 7\n" << mat << std::endl;
            Mtm.noalias()   = Gb2b1 - Tb2b1;
            std::cout << "mat = 8\n" << mat << std::endl;
            Mml.noalias()   = (Tb1b2 - Gb1b2)*diag(Hb2b2);
            std::cout << "mat = 9\n" << mat << std::endl;
            Mmm.noalias()   = Gb1b1 - Tb1b1;
            Mmm.diagonal() -= inv(Hb1b1);
            std::cout << "mat = 10\n" << mat << std::endl;
        }
        else
        {
            Mtl.noalias()   = Tb2b2 * diag(Hb2b2);
            Mtl.diagonal() += ones(nbx2);
            Mtm.noalias()   = -Tb2b1;
            Mml.noalias()   = Tb1b2 * diag(Hb2b2);
            Mmm.noalias()   = -Tb1b1;
            Mmm.diagonal() -= inv(Hb1b1);
        }

        Mtr.noalias() = Sb2n2;
        std::cout << "mat = 11\n" << mat << std::endl;
        Mmr.noalias() = Sb1n2;
        std::cout << "mat = 12\n" << mat << std::endl;
        Mbl.noalias() = -tr(Sb2n2) * diag(Hb2b2);
        std::cout << "mat = 13\n" << mat << std::endl;
        Mbm.noalias() = tr(Sb1n2);
        std::cout << "mat = 14\n" << mat << std::endl;
        Mbr           = diag(Hn2n2);
        std::cout << "mat = 15\n" << mat << std::endl;

        luxb.compute(M);
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDiagonal(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        auto Sbnx  = canonicalizer.S().topLeftCorner(nbx, nnx);
        auto Sbnf  = canonicalizer.S().topRightCorner(nbx, nnf);
        auto Sb1n1 = Sbnx.bottomRightCorner(nbx1, nnx1);
        auto Sb2n1 = Sbnx.topRightCorner(nbx2, nnx1);
        auto Sb2n2 = Sbnx.topLeftCorner(nbx2, nnx2);

        auto R = canonicalizer.R().topRows(nbx);

        auto H     = mat.col(0).head(nx);
        auto Hbb   = H.head(nbx);
        auto Hnn   = H.tail(nnx);
        auto Hb1b1 = Hbb.tail(nbx1);
        auto Hb2b2 = Hbb.head(nbx2);
        auto Hn1n1 = Hnn.tail(nnx1);
        auto Hn2n2 = Hnn.head(nnx2);

        auto G     = mat.bottomLeftCorner(nbx, nbx);
        auto Gb2b2 = G.topLeftCorner(nbx2, nbx2);
        auto Gb1b2 = G.bottomLeftCorner(nbx1, nbx2);

        auto a   = vec.head(n);
        auto ax  = a.head(nx);
        auto af  = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto anf = af.tail(nnf);
        auto ab1 = abx.tail(nbx1);
        auto ab2 = abx.head(nbx2);
        auto an1 = anx.tail(nnx1);
        auto an2 = anx.head(nnx2);

        auto b   = vec.tail(m);
        auto bb  = b.head(nbx);
        auto bb1 = bb.tail(nbx1);
        auto bb2 = bb.head(nbx2);

        auto r = x.head(nnx2 + nbx);

        auto xb2 = r.head(nbx2);
        auto yb1 = r.segment(nbx2, nbx1);
        auto xn2 = r.tail(nnx2);

        a.noalias() = rows(rhs.a(), iordering);
        bb.noalias() = R*rhs.b() - Sbnf*anf;

        an1.noalias() -= tr(Sb2n1) * ab2;
        an1.noalias()  = an1/Hn1n1;
        an2.noalias() -= tr(Sb2n2) * ab2;
        bb1.noalias() -= ab1/Hb1b1 - Sb1n1*an1;
        bb2.noalias() -= Sb2n1*an1;

        if(lhs.G().size())
        {
            bb1.noalias() -= Gb1b2 * ab2;
            bb2.noalias() -= Gb2b2 * ab2;
        }

        r << bb2, bb1, an2;

        r.noalias() = luxb.solve(r);

        ab1.noalias() = (ab1 - yb1)/Hb1b1;
        bb2.noalias() = (ab2 - Hb2b2 % xb2);
        an1.noalias() -= (tr(Sb1n1)*yb1 + tr(Sb2n1) * (bb2 - ab2))/Hn1n1;

        an2.noalias() = xn2;
        bb1.noalias() = yb1;
        ab2.noalias() = xb2;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R)*bb;

        // Permute back the variables `x` to their original ordering
        rows(x, iordering).noalias() = a;

        std::cout << "r   = " << tr(r) << std::endl;
        std::cout << "xn2 = " << tr(xn2) << std::endl;
        std::cout << "yb1 = " << tr(yb1) << std::endl;
        std::cout << "xb2 = " << tr(xb2) << std::endl;
        std::cout << "xb1 = " << tr(ab1) << std::endl;
        std::cout << "yb2 = " << tr(bb2) << std::endl;
        std::cout << "xn1 = " << tr(an1) << std::endl;
        std::cout << "x   = " << tr(x) << std::endl;
        std::cout << "y   = " << tr(y) << std::endl;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspaceZeroG(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        auto Sbnx = canonicalizer.S().topLeftCorner(nbx, nnx);

        // Create auxiliary matrix views
        auto Hx   = mat.topLeftCorner(nx, nx);
        auto B    = mat.bottomLeftCorner(m, n);
        auto Hbbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbnx = Hx.topRightCorner(nbx, nnx);
        auto Hnbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnnx = Hx.bottomRightCorner(nnx, nnx);
        auto Bbnx = B.topLeftCorner(nbx, nnx);
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
        if(nnx) luxn.compute(Mnnx);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspaceZeroG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        auto Sbnx = canonicalizer.S().topLeftCorner(nbx, nnx);
        auto Sbnf = canonicalizer.S().topRightCorner(nbx, nnf);

        auto R = canonicalizer.R().topRows(nbx);

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
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to values of `b` for free basic variables.
        auto bb = vec.segment(nx, nbx);

        // The view in `y` corresponding to values of `y` for free basic variables.
        auto yb = y.head(nbx);

        // Create auxiliary matrix views
        auto Hx   = mat.topLeftCorner(nx, nx);
        auto Hbbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbnx = Hx.topRightCorner(nbx, nnx);
        auto Hnbx = Hx.bottomLeftCorner(nnx, nbx);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = rows(a, ivx);
        af.noalias() = rows(a, ivf);

        // Set the vector `bx`
        bb.noalias() = R*b - Sbnf*anf;

        auto xx = x.head(nx);
        auto xbx = xx.head(nbx);
        auto xnx = xx.tail(nnx);

        // Compute the saddle point problem solution
        yb.noalias()  = abx - Hbbx*bb;
        xnx.noalias() = anx - Hnbx*bb - tr(Sbnx)*yb;
        if(nnx) anx.noalias() = luxn.solve(xnx);
        xbx.noalias() = abx;
        abx.noalias() = bb - Sbnx*anx;
        bb.noalias()  = xbx - Hbbx*abx - Hbnx*anx;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R)*bb;

        // Permute back the variables `x` to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspaceDenseG(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the Jacobian matrix A
        updateCanonicalForm(lhs);

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R().topRows(nbx);

        // The indices of the free variables
        auto ivx  = iordering.head(nx);

        auto Sbnx = S.topLeftCorner(nbx, nnx);

        // Create auxiliary matrix views
        auto Hx = mat.topLeftCorner(nx, nx);

        Hx.noalias() = submatrix(lhs.H(), ivx, ivx);

        auto Hbx  = mat.topRightCorner(nx, nbx);
        auto Hbbx = Hbx.topRows(nbx);
        auto Hnbx = Hbx.bottomRows(nnx);

        Hbbx.noalias() = Hx.topLeftCorner(nbx, nbx);
        Hnbx.noalias() = Hx.bottomLeftCorner(nnx, nbx);

        auto Gbb  = mat.bottomRightCorner(nb, nb);
        auto Gbbx = Gbb.topLeftCorner(nbx, nbx);

        Gbbx = R * lhs.G() * tr(R);

        auto M    = mat.topLeftCorner(nx, nx);
        auto Mbbx = M.topLeftCorner(nbx, nbx);
        auto Mbnx = M.topRightCorner(nbx, nnx);
        auto Mnbx = M.bottomLeftCorner(nnx, nbx);
        auto Mnnx = M.bottomRightCorner(nnx, nnx);

        Mbbx.noalias()   = -Hbbx*Gbbx;
        Mbbx.diagonal() += ones(nbx);
        Mnbx.noalias()   = tr(Sbnx) - Hnbx*Gbbx;
        Mbnx.noalias()  -= Hbbx*Sbnx;
        Mnnx.noalias()  -= Hnbx*Sbnx;

        // Compute the LU decomposition of Mnnx.
        if(nx) luxn.compute(M);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspaceDenseG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to members of the saddle point right-hand side vector.
        auto a = rhs.a();
        auto b = rhs.b();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

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
        auto bb  = vec.segment(nx, nb);
        auto bbx = bb.head(nbx);

//        // The view in `y` corresponding to values of `y` for free basic variables.
//        auto yx = y.head(nbx);

        // Create auxiliary matrix views
        auto Hbx  = mat.topRightCorner(nx, nbx);
        auto Hbbx = Hbx.topRows(nbx);
        auto Hnbx = Hbx.bottomRows(nnx);
        auto Gbb  = mat.bottomRightCorner(nb, nb);
        auto Gbbx = Gbb.topLeftCorner(nbx, nbx);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = rows(a, ivx);
        af.noalias() = rows(a, ivf);

        // Set the vector `bx`
        bb.noalias() = R*b - Ibf*abf - Sbnf*anf;

        abx.noalias() -= Hbbx*bbx;
        anx.noalias() -= Hnbx*bbx;

        if(nx) ax.noalias() = luxn.solve(ax);

        y.noalias() = tr(Rx)*abx;

        bbx.noalias() -= Sbnx*anx + Gbbx*abx;
        abx.noalias() = bbx;

//        auto xx = x.head(nx);
//        auto xbx = xx.head(nbx);
//        auto xnx = xx.tail(nnx);
//
//        // Compute the saddle point problem solution
//        yx.noalias()  = abx - Hbbx*bx;
//        xnx.noalias() = anx - Hnbx*bx - tr(Sbnx)*yx;
//        if(nnx) anx.noalias() = luxn.solve(xnx);
//        xbx.noalias() = abx;
//        abx.noalias() = bx - Sbnx*anx;
//        bx.noalias()  = xbx - Hbbx*abx - Hbnx*anx;
//
//        // Compute the y vector without canonicalization
//        y.noalias() = tr(Rx)*bx;

        // Permute back the variables `x` to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspace(SaddlePointMatrix lhs) -> void
    {
        if(lhs.G().size()) decomposeNullspaceDenseG(lhs);
        else decomposeNullspaceZeroG(lhs);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspace(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        if(lhs.G().size()) solveNullspaceDenseG(lhs, rhs, sol);
        else solveNullspaceZeroG(lhs, rhs, sol);
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointMatrix lhs) -> SaddlePointResult
    {
        SaddlePointResult res;
        switch(options.method)
        {
        case SaddlePointMethod::PartialPivLU:
        case SaddlePointMethod::FullPivLU: decomposeLU(lhs); break;
        case SaddlePointMethod::RangespaceDiagonal: decomposeRangespaceDiagonal(lhs); break;
        default: decomposeNullspace(lhs); break;
        }
        return res.stop();
    }

    /// Solve the saddle point problem with diagonal Hessian matrix.
    auto solve(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> SaddlePointResult
    {
        SaddlePointResult res;
        switch(options.method)
        {
        case SaddlePointMethod::PartialPivLU:
        case SaddlePointMethod::FullPivLU: solveLU(lhs, rhs, sol); break;
        case SaddlePointMethod::RangespaceDiagonal: solveRangespaceDiagonal(lhs, rhs, sol); break;
        default: solveNullspace(lhs, rhs, sol); break;
        }
        return res.stop();
    }

//    auto solve(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> SaddlePointResult
//    {
//        // The result of this method call
//        SaddlePointResult res;
//
//        // Assemble the saddle point coefficient matrix and the right-hand side vector
//        mat << lhs;
//        vec << rhs;
//
//        // Solve the saddle point problem
//        vec.noalias() = mat.lu().solve(vec);
//
//        // Set the saddle point solution with values in vec
//        sol = vec;
//
//        return res.stop();
//    }
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

auto SaddlePointSolver::setMethodMoreAccurate(SaddlePointMatrix lhs, SaddlePointVector rhs) -> void
{
}

auto SaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->options;
}

auto SaddlePointSolver::canonicalize(MatrixXdConstRef A) -> SaddlePointResult
{
    return pimpl->canonicalize(A);
}

auto SaddlePointSolver::decompose(SaddlePointMatrix lhs) -> SaddlePointResult
{
    return pimpl->decompose(lhs);
}

auto SaddlePointSolver::solve(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution sol) -> SaddlePointResult
{
    return pimpl->solve(lhs, rhs, sol);
}

auto SaddlePointSolver::update(VectorXiConstRef ordering) -> void
{
    pimpl->canonicalizer.update(ordering);
}

} // namespace Optima

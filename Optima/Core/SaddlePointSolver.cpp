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

    /// The number of basic and non-basic variables.
    Index nb, nn;

    /// The number of linearly dependent rows in *A*
    Index nl;

    /// The number of free and fixed variables.
    Index nx, nf;

    /// The number of free and fixed basic variables.
    Index nbx, nbf;

    /// The number of free and fixed non-basic variables.
    Index nnx, nnf;

    /// The number of pivot free basic variables.
    Index nb1, nn1;

    /// The number of non-pivot free non-basic variables.
    Index nb2, nn2;

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

        // Set the number of linearly dependent rows in A
        nl = m - nb;

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
        nb1 = nbx;
        nn1 = nnx;

        // Set the number of non-pivot free non-basic variables.
        nb2 = 0;
        nn2 = 0;

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
        nb2 = 0; while(nb2 < nbx && weights[ibasic[nb2]] > 1.0) ++nb2;

        // Update the number of non-pivot free non-basic variables.
        nn2 = 0; while(nn2 < nnx && weights[inonbasic[nn2]] > norminf(S.col(nn2))) ++nn2;

        // Update the number of pivot free basic and non-basic variables.
        nb1 = nbx - nb2;
        nn1 = nnx - nn2;

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
        // Update the canonical form of the matrix A
        updateCanonicalForm(lhs);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sbxnx = canonicalizer.S().topLeftCorner(nbx, nnx);
        auto Ibxbx = identity(nbx, nbx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(nx + nbx, nx + nbx);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(nbx, nbx).noalias() = Ibxbx;
        M.topRightCorner(nbx, nbx).noalias() = Ibxbx;

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.bottomRows(nbx).middleCols(nbx, nnx) = Sbxnx;
        M.rightCols(nbx).middleRows(nbx, nnx)  = tr(Sbxnx);

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

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU_ZeroG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Alias to the matrices of the canonicalization process
        auto Sbn   = canonicalizer.S();
        auto Sbxnf = Sbn.topRightCorner(nbx, nnf);
        auto Sbfnf = Sbn.bottomRightCorner(nbf, nnf);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to values of `b` for linerly independent equation.
        auto b   = vec.segment(nx, m);
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);

        ax.noalias() = rows(rhs.a(), ivx);
        af.noalias() = rows(rhs.a(), ivf);
        b.noalias()  = R * rhs.b();

        bbx -= Sbxnf * anf;
        bbf -= Sbfnf * anf + abf;

        auto r = vec.head(nx + nbx);

        std::cout << "r = " << tr(r) << std::endl;

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            r.noalias() = luxy_partial.solve(r);
        else
            r.noalias() = luxy_full.solve(r);

        // Compute the `y` vector without canonicalization
        y.noalias() = tr(R)*b;

        // Permute back the variables x to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeLU_DenseG(SaddlePointMatrix lhs) -> void
    {
        // Update the canonical form of the matrix A
        updateCanonicalForm(lhs);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sbn   = canonicalizer.S();
        auto Sbxnx = Sbn.topLeftCorner(nbx, nnx);

        auto R = canonicalizer.R();

        auto Ibxbx = identity(nbx, nbx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(m + nx, m + nx);

        M.topLeftCorner(nx, nx) = submatrix(lhs.H(), ivx, ivx);

        M.middleCols(nx, nbx).topRows(nx) << Ibxbx, tr(Sbxnx);
        M.middleRows(nx, nbx).leftCols(nx) << Ibxbx, Sbxnx;

        M.topRightCorner(nf + nl, nf + nl).setZero();
        M.bottomLeftCorner(nf + nl, nf + nl).setZero();

        M.bottomRightCorner(m, m) = R * lhs.G() * tr(R);

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            luxy_partial.compute(M);
        else
            luxy_full.compute(M);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveLU_DenseG(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Alias to the matrices of the canonicalization process
        auto Sbn   = canonicalizer.S();
        auto Sbxnf = Sbn.topRightCorner(nbx, nnf);
        auto Sbfnf = Sbn.bottomRightCorner(nbf, nnf);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The view in `vec` corresponding to values of `a` for free and fixed variables.
        auto ax = vec.head(nx);
        auto af = vec.tail(nf);

        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // The view in `vec` corresponding to values of `b` for linerly independent equation.
        auto b   = vec.segment(nx, m);
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);

        ax.noalias() = rows(rhs.a(), ivx);
        af.noalias() = rows(rhs.a(), ivf);
        b.noalias() = R * rhs.b();

        bbx -= Sbxnf * anf;
        bbf -= Sbfnf * anf + abf;

        auto r = vec.head(nx + m);

        std::cout << "r = " << tr(r) << std::endl;

        // Compute the LU decomposition of M.
        if(options.method == SaddlePointMethod::PartialPivLU)
            r.noalias() = luxy_partial.solve(r);
        else
            r.noalias() = luxy_full.solve(r);

        // Compute the `y` vector without canonicalization
        y.noalias() = tr(R)*b;

        // Permute back the variables x to their original ordering
        rows(x, ivx).noalias() = ax;
        rows(x, ivf).noalias() = af;
    }

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
        // Update the canonical form of the matrix A
        updateCanonicalForm(lhs);

        // The indices of the free variables, basic variables and non-basic variables
        auto ivx = iordering.head(nx);

        // The canonicalizer matrix R such that R*A*Q = [Ibb Sbn]
        auto R = canonicalizer.R().topRows(nbx);

        // The matrix Sbn of the canonicalization of A and its submatrices
        auto Sbn   = canonicalizer.S();
        auto Sbxnx = Sbn.topLeftCorner(nbx, nnx);
        auto Sb1n2 = Sbxnx.bottomLeftCorner(nb1, nn2);
        auto Sb2n2 = Sbxnx.topLeftCorner(nb2, nn2);
        auto Sbxn1 = Sbxnx.rightCols(nn1);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = mat.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);
        auto Hn2n2 = Hnxnx.head(nn2);

        // The matrix G' = R * G * tr(R)
        auto G     = mat.bottomLeftCorner(m, m);
        auto Gb1   = G.leftCols(nb1);
        auto Gb2   = G.middleCols(nb1, nb2);
        auto Gbf   = G.middleCols(nb1 + nb2, nbf);
        auto Gl    = G.rightCols(nl);
        auto Gb1b1 = Gb1.topRows(nb1);
        auto Gb2b1 = Gb1.middleRows(nb1, nb2);
        auto Gbfb1 = Gb1.middleRows(nb1 + nb2, nbf);
        auto  Glb1 = Gb1.bottomRows(nl);
        auto Gb1b2 = Gb2.topRows(nb1);
        auto Gb2b2 = Gb2.middleRows(nb1, nb2);
        auto Gbfb2 = Gb2.middleRows(nb1 + nb2, nbf);
        auto  Glb2 = Gb2.bottomRows(nl);
        auto Gb1bf = Gbf.topRows(nb1);
        auto Gb2bf = Gbf.middleRows(nb1, nb2);
        auto Gbfbf = Gbf.middleRows(nb1 + nb2, nbf);
        auto  Glbf = Gbf.bottomRows(nl);
        auto  Gb1l = Gl.topRows(nb1);
        auto  Gb2l = Gl.middleRows(nb1, nb2);
        auto  Gbfl = Gl.middleRows(nb1 + nb2, nbf);
        auto   Gll = Gl.bottomRows(nl);

        // The auxiliary matrix Tbxbx = Sbxn1 * Bn1bx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tb2b2 = Tbxbx.topLeftCorner(nb2, nb2);
        auto Tb2b1 = Tbxbx.topRightCorner(nb2, nb1);
        auto Tb1b2 = Tbxbx.bottomLeftCorner(nb1, nb2);
        auto Tb1b1 = Tbxbx.bottomRightCorner(nb1, nb1);

        // The auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        auto Bn1bx = mat.rightCols(nbx).middleRows(nbx, nn1);

        // The matrix M of the system of linear equations
        auto M     = mat.bottomRightCorner(m + nn2, m + nn2);
        auto Mn2   = M.leftCols(nn2);
        auto Mb1   = M.middleCols(nn2, nb1);
        auto Mb2   = M.middleCols(nn2 + nb1, nb2);
        auto Mbf   = M.middleCols(nn2 + nb1 + nb2, nbf);
        auto Ml    = M.rightCols(nl);
        auto Mn2n2 = Mn2.topRows(nn2);
        auto Mb1n2 = Mn2.middleRows(nn2, nb1);
        auto Mb2n2 = Mn2.middleRows(nn2 + nb1, nb2);
        auto Mbfn2 = Mn2.middleRows(nn2 + nb1 + nb2, nbf);
        auto  Mln2 = Mn2.bottomRows(nl);
        auto Mn2b1 = Mb1.topRows(nn2);
        auto Mb1b1 = Mb1.middleRows(nn2, nb1);
        auto Mb2b1 = Mb1.middleRows(nn2 + nb1, nb2);
        auto Mbfb1 = Mb1.middleRows(nn2 + nb1 + nb2, nbf);
        auto  Mlb1 = Mb1.bottomRows(nl);
        auto Mn2b2 = Mb2.topRows(nn2);
        auto Mb1b2 = Mb2.middleRows(nn2, nb1);
        auto Mb2b2 = Mb2.middleRows(nn2 + nb1, nb2);
        auto Mbfb2 = Mb2.middleRows(nn2 + nb1 + nb2, nbf);
        auto  Mlb2 = Mb2.bottomRows(nl);
        auto Mn2bf = Mbf.topRows(nn2);
        auto Mb1bf = Mbf.middleRows(nn2, nb1);
        auto Mb2bf = Mbf.middleRows(nn2 + nb1, nb2);
        auto Mbfbf = Mbf.middleRows(nn2 + nb1 + nb2, nbf);
        auto  Mlbf = Mbf.bottomRows(nl);
        auto  Mn2l = Ml.topRows(nn2);
        auto  Mb1l = Ml.middleRows(nn2, nb1);
        auto  Mb2l = Ml.middleRows(nn2 + nb1, nb2);
        auto  Mbfl = Ml.middleRows(nn2 + nb1 + nb2, nbf);
        auto   Mll = Ml.bottomRows(nl);

        // Setting Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        Hx.noalias() = rows(lhs.H().diagonal(), ivx);

        // Computing G' = R * G * tr(R)
        G.noalias() = R * lhs.G() * tr(R);

        // Computing the auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        Bn1bx.noalias() = diag(inv(Hn1n1)) * tr(Sbxn1);

        // Computing the auxiliary matrix Tbxbx = Sbxn1 * Bn1bx
        Tbxbx.noalias() = Sbxn1 * Bn1bx;

        // Setting the columns of M with dimension nn2
        Mn2n2           = diag(Hn2n2);
        Mb1n2.noalias() = Sb1n2;
        Mb2n2.noalias() = Sb2n2;
        Mbfn2.setZero();
        Mln2.setZero();

        // Setting the columns of M with dimension nb1
        Mn2b1.noalias()   = tr(Sb1n2);
        Mb1b1.noalias()   = Gb1b1 - Tb1b1;
        Mb1b1.diagonal() -= inv(Hb1b1);
        Mb2b1.noalias()   = Gb2b1 - Tb2b1;
        Mbfb1.noalias()   = Gbfb1;
        Mlb1.noalias()    = Glb1;

        // Setting the columns of M with dimension nb2
        Mn2b2.noalias()   = -tr(Sb2n2) * Hb2b2;
        Mb1b2.noalias()   = (Tb1b2 - Gb1b2)*diag(Hb2b2);
        Mb2b2.noalias()   = (Tb2b2 - Gb2b2)*diag(Hb2b2);
        Mb2b2.diagonal() += ones(nb2);
        Mbfb2.noalias()   = -Gbfb2 * diag(Hb2b2);
        Mlb2.noalias()    =  -Glb2 * diag(Hb2b2);

        // Setting the columns of M with dimension nbf
        Mn2bf.setZero();
        Mb1bf.noalias() = Gb1bf;
        Mb2bf.noalias() = Gb2bf;
        Mbfbf.noalias() = Gbfbf;
         Mlbf.noalias() =  Glbf;

        // Setting the columns of M with dimension nl
        Mn2l.setZero();
        Mb1l.noalias() = Gb1l;
        Mb2l.noalias() = Gb2l;
        Mbfl.noalias() = Gbfl;
         Mll.noalias() =  Gll;

        // Computing the LU decomposition of matrix M
        luxb.compute(M);
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDiagonal(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution& sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x();
        auto y = sol.y();

        // The canonicalizer matrix R such that R*A*Q = [Ibb Sbn]
        auto R = canonicalizer.R().topRows(nbx);

        // The matrix Sbn of the canonicalization of A and its submatrices
        auto Sbn   = canonicalizer.S();
        auto Sbxnx = Sbn.topLeftCorner(nbx, nnx);
        auto Sbxnf = Sbn.topRightCorner(nbx, nnf);
        auto Sbfnf = Sbn.bottomRightCorner(nbf, nnf);
        auto Sb1n1 = Sbxnx.bottomRightCorner(nb1, nn1);
        auto Sb2n1 = Sbxnx.topRightCorner(nb2, nn1);
        auto Sb2nx = Sbxnx.topRows(nb2);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = mat.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);

        // The matrix G' = R * G * tr(R)
        auto G     = mat.bottomLeftCorner(m, m);
        auto Gb2   = G.middleCols(nb1, nb2);
        auto Gb1b2 = Gb2.topRows(nb1);
        auto Gb2b2 = Gb2.middleRows(nb1, nb2);
        auto Gbfb2 = Gb2.middleRows(nb1 + nb2, nbf);
        auto  Glb2 = Gb2.bottomRows(nl);

        auto a   = vec.head(n);
        auto ax  = a.head(nx);
        auto af  = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);
        auto ab1 = abx.tail(nb1);
        auto ab2 = abx.head(nb2);
        auto an1 = anx.tail(nn1);
        auto an2 = anx.head(nn2);

        auto b   = vec.tail(m);
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);
        auto bl  = b.tail(nl);
        auto bb1 = bbx.tail(nb1);
        auto bb2 = bbx.head(nb2);

        b = R * rhs.b();
        bbx -= Sbxnf * anf;
        bbf -= Sbfnf * anf + abf;

        a = rows(rhs.a(), iordering);
        anx -= tr(Sb2nx) * ab2;

        an1.noalias() = an1/Hn1n1;

        bb1 -= ab1/Hb1b1;
        bb1 -= Sb1n1 * an1;
        bb1 -= Gb1b2 * ab2;

        bb2 -= Sb2n1 * an1;
        bb2 -= Gb2b2 * ab2;
        bbf -= Gbfb2 * ab2;
        bl  -=  Glb2 * ab2;

        auto r = x.head(nn2 + m);

        auto xn2 = r.head(nn2);
        auto yb1 = r.segment(nn2, nb1);
        auto xb2 = r.segment(nn2 + nb1, nb2);
        auto ybf = r.segment(nn2 + nb1 + nb2, nbf);
        auto yl = r.tail(nl);

        r << an2, bb1, bb2, bbf, bl;

        r.noalias() = luxb.solve(r);

        ab1.noalias() = (ab1 - yb1)/Hb1b1;
        bb2.noalias() = (ab2 - Hb2b2 % xb2);
        an1.noalias() -= (tr(Sb1n1)*yb1 + tr(Sb2n1)*(bb2 - ab2))/Hn1n1;

        an2.noalias() = xn2;
        bb1.noalias() = yb1;
        ab2.noalias() = xb2;
        bbf.noalias() = ybf;
         bl.noalias() = yl;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R) * b;

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
        // Update the canonical form of the matrix A
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
        // Update the canonical form of the matrix A
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

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

#include "SaddlePointSolver.hpp"

// Eigen includes
#include <eigen3/Eigenx/Core>
#include <eigen3/Eigen/LU>
using namespace Eigen;

// Optima includes
#include <Optima/Canonicalizer.hpp>
#include <Optima/Exception.hpp>
#include <Optima/Result.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/Utils.hpp>
#include <Optima/VariantMatrix.hpp>

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
    Vector weights;

    /// The 'H' matrix in the saddle point matrix.
    VariantMatrix H;

    /// The 'G' matrix in the saddle point matrix.
    VariantMatrix G;

    /// The workspace for the right-hand side vectors a and b
    Vector a, b;

    /// The matrix used as a workspace for the decompose and solve methods.
    Matrix mat;

    /// The vector used as a workspace for the decompose and solve methods.
    Vector vec;

    /// The ordering of the variables as (free-basic, free-non-basic, fixed-basic, fixed-non-basic)
    VectorXi iordering;

    /// The LU decomposition solver.
    Eigen::PartialPivLU<Matrix> lu;

    /// The boolean flag that indicates that the decomposed saddle point matrix was degenerate with no free variables.
    bool degenerate = false;

    /// Update the order of the variables.
    auto reorderVariables(VectorXiConstRef ordering) -> void
    {
        // Update the ordering of the canonicalizer object
        canonicalizer.updateWithNewOrdering(ordering);
    }

    /// Canonicalize the coefficient matrix *A* of the saddle point problem.
    auto initialize(MatrixConstRef A) -> Result
    {
        // The result of this method call
        Result res;

        // Set the number of rows and columns in A
        m = A.rows();
        n = A.cols();

        // Allocate auxiliary memory
        a.resize(n);
        b.resize(m);
        mat.resize(n + m, n + m);
        vec.resize(n + m);
        weights.resize(n);
        iordering.resize(n);

        // Compute the canonical form of matrix A
        canonicalizer.compute(A);

        // Set the number of basic and non-basic variables
        nb = canonicalizer.numBasicVariables();
        nn = canonicalizer.numNonBasicVariables();

        // Set the number of linearly dependent rows in A
        nl = m - nb;

        // Set the number of free and fixed variables
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
        iordering.head(nb) = canonicalizer.indicesBasicVariables();
        iordering.tail(nn) = canonicalizer.indicesNonBasicVariables();

        return res.stop();
    }

    /// Update the canonical form of the coefficient matrix *A* of the saddle point problem.
    auto updateCanonicalForm(SaddlePointMatrix lhs) -> void
    {
        // Update the number of fixed and free variables
        nf = lhs.nf;
        nx = n - nf;

        // Determine if the saddle point matrix is degenerate
        degenerate = nx == 0;

        // Skip the rest if there is no free variables
        if(degenerate)
            return;

        // The diagonal entries of the Hessian matrix
        auto Hdd = lhs.H.diagonalRef();

        // The diagonal entries of the Hessian matrix corresponding to free variables
        auto Hxx = Hdd.head(nx);

        // Update the priority weights for the update of the canonical form
        weights.head(nx).noalias() = abs(inv(Hxx));
        weights.tail(nf).noalias() = -linspace(nf, 1, nf);

        // Update the canonical form and the ordering of the variables
        canonicalizer.updateWithPriorityWeights(weights);

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = canonicalizer.indicesBasicVariables();
        const auto inonbasic = canonicalizer.indicesNonBasicVariables();

        // Get the S matrix of the canonical form of A
        const auto S = canonicalizer.S();

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

        // Update the ordering of the free variables as xx = [xbx xnx]
        iordering.head(nx).head(nbx) = ibasic.head(nbx);
        iordering.head(nx).tail(nnx) = inonbasic.head(nnx);

        // Update the ordering of the fixed variables as xf = [xbf xnf]
        iordering.tail(nf).head(nbf) = ibasic.tail(nbf);
        iordering.tail(nf).tail(nnf) = inonbasic.tail(nnf);
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointMatrix lhs) -> Result
    {
        Result res;










        canonicalizer.compute(lhs.A);










        // Update the canonical form of the matrix A
        updateCanonicalForm(lhs);

        // Check if the saddle point matrix is degenerate, with no free variables.
        if(degenerate)
            decomposeDegenerateCase(lhs);

        else switch(options.method)
        {
        case SaddlePointMethod::Nullspace: decomposeNullspace(lhs); break;
        case SaddlePointMethod::Rangespace: decomposeRangespace(lhs); break;
        default: decomposeFullspace(lhs); break;
        }

        return res.stop();
    }

    /// Decompose the saddle point matrix for the degenerate case of no free variables.
    auto decomposeDegenerateCase(SaddlePointMatrix lhs) -> void
    {
        if(lhs.G.structure == MatrixStructure::Dense)
        {
            // Set the G matrix to dense structure
            G.setDense(m);

            // Set the G matrix from the given saddle point matrix
            G.dense << lhs.G;

            // Compute the LU decomposition of G.
            lu.compute(G.dense);
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspace(SaddlePointMatrix lhs) -> void
    {
        switch(lhs.G.structure) {
            case MatrixStructure::Zero: decomposeFullspaceZeroG(lhs); break;
            default: decomposeFullspaceDenseG(lhs); break;
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspaceZeroG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a dense structure
        H.setDense(n);

        // Set the G matrix to zero structure
        G.setZero();

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
        M.topLeftCorner(nx, nx) << lhs.H(ivx);

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspaceDenseG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a dense structure
        H.setDense(n);

        // Set the G matrix to dense structure
        G.setDense(m);

        // Set the G matrix from the given saddle point matrix
        G.dense << lhs.G;

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sbn   = canonicalizer.S();
        auto Sbxnx = Sbn.topLeftCorner(nbx, nnx);

        auto R = canonicalizer.R();

        auto Ibxbx = identity(nbx, nbx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(m + nx, m + nx);

        M.topLeftCorner(nx, nx) << lhs.H(ivx);
        M.middleCols(nx, nbx).topRows(nx) << Ibxbx, tr(Sbxnx);
        M.middleRows(nx, nbx).leftCols(nx) << Ibxbx, Sbxnx;
        M.topRightCorner(nx, nbf + nl).setZero();
        M.bottomLeftCorner(nbf + nl, nx).setZero();
        M.bottomRightCorner(m, m) = R * G.dense * tr(R);

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceAux(SaddlePointMatrix lhs) -> void
    {
        switch(lhs.G.structure) {
            case MatrixStructure::Zero: decomposeRangespaceZeroG(lhs); break;
            default: decomposeRangespaceDenseG(lhs); break;
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespace(SaddlePointMatrix lhs) -> void
    {
        switch(lhs.H.structure) {
        case MatrixStructure::Dense: decomposeNullspace(lhs); break;
        default: decomposeRangespaceAux(lhs); break;
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceZeroG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a diagonal structure
        H.setDiagonal(n);

        // Set the G matrix to zero structure
        G.setZero();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sb1n2 = Sbxnx.bottomLeftCorner(nb1, nn2);
        auto Sb2n2 = Sbxnx.topLeftCorner(nb2, nn2);
        auto Sbxn1 = Sbxnx.rightCols(nn1);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = H.diagonal.head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);
        auto Hn2n2 = Hnxnx.head(nn2);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx.noalias() = lhs.H.diagonal(ivx);

        // The auxiliary matrix Tbxbx = Sbxn1 * Bn1bx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tb2b2 = Tbxbx.topLeftCorner(nb2, nb2);
        auto Tb2b1 = Tbxbx.topRightCorner(nb2, nb1);
        auto Tb1b2 = Tbxbx.bottomLeftCorner(nb1, nb2);
        auto Tb1b1 = Tbxbx.bottomRightCorner(nb1, nb1);

        // The auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        auto Bn1bx = mat.rightCols(nbx).middleRows(nbx, nn1);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(nb1 + nb2 + nn2, nb1 + nb2 + nn2);

        auto Mn2 = M.topRows(nn2);
        auto Mb1 = M.middleRows(nn2, nb1);
        auto Mb2 = M.middleRows(nn2 + nb1, nb2);

        auto Mn2n2 = Mn2.leftCols(nn2);
        auto Mn2b1 = Mn2.middleCols(nn2, nb1);
        auto Mn2b2 = Mn2.middleCols(nn2 + nb1, nb2);

        auto Mb1n2 = Mb1.leftCols(nn2);
        auto Mb1b1 = Mb1.middleCols(nn2, nb1);
        auto Mb1b2 = Mb1.middleCols(nn2 + nb1, nb2);

        auto Mb2n2 = Mb2.leftCols(nn2);
        auto Mb2b1 = Mb2.middleCols(nn2, nb1);
        auto Mb2b2 = Mb2.middleCols(nn2 + nb1, nb2);

        // Computing the auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        Bn1bx.noalias() = diag(inv(Hn1n1)) * tr(Sbxn1);

        // Computing the auxiliary matrix Tbxbx = Sbxn1 * Bn1bx
        Tbxbx.noalias() = Sbxn1 * Bn1bx;

        // Setting the columns of M with dimension nn2
        Mn2n2           = diag(Hn2n2);
        Mb1n2.noalias() = Sb1n2;
        Mb2n2.noalias() = Sb2n2;

        // Setting the columns of M with dimension nb1
        Mn2b1.noalias()   = tr(Sb1n2);
        Mb1b1.noalias()   = -Tb1b1;
        Mb1b1.diagonal() -= inv(Hb1b1);
        Mb2b1.noalias()   = -Tb2b1;

        // Setting the columns of M with dimension nb2
        Mn2b2.noalias()   = -tr(Sb2n2) * diag(Hb2b2);
        Mb1b2.noalias()   = Tb1b2*diag(Hb2b2);
        Mb2b2.noalias()   = Tb2b2*diag(Hb2b2);
        Mb2b2.diagonal() += ones(nb2);

        // Computing the LU decomposition of matrix M
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceDenseG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a diagonal structure
        H.setDiagonal(n);

        // Set the G matrix to dense structure
        G.setDense(m);

        // Set the G matrix from the given saddle point matrix
        G.dense << lhs.G;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sb1n2 = Sbxnx.bottomLeftCorner(nb1, nn2);
        auto Sb2n2 = Sbxnx.topLeftCorner(nb2, nn2);
        auto Sbxn1 = Sbxnx.rightCols(nn1);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = H.diagonal.head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);
        auto Hn2n2 = Hnxnx.head(nn2);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.dense.topRows(nbx);
        auto Gbf   = G.dense.middleRows(nbx, nbf);
        auto Gbl   = G.dense.bottomRows(nl);
        auto Gbxbx = Gbx.leftCols(nbx);
        auto Gbxbf = Gbx.middleCols(nbx, nbf);
        auto Gbxbl = Gbx.rightCols(nl);
        auto Gbfbx = Gbf.leftCols(nbx);
        auto Gbfbf = Gbf.middleCols(nbx, nbf);
        auto Gbfbl = Gbf.rightCols(nl);
        auto Gblbx = Gbl.leftCols(nbx);
        auto Gblbf = Gbl.middleCols(nbx, nbf);
        auto Gblbl = Gbl.rightCols(nl);

        auto Gb1b1 = Gbxbx.bottomRightCorner(nb1, nb1);
        auto Gb1b2 = Gbxbx.bottomLeftCorner(nb1, nb2);
        auto Gb2b1 = Gbxbx.topRightCorner(nb2, nb1);
        auto Gb2b2 = Gbxbx.topLeftCorner(nb2, nb2);

        auto Gb1bf = Gbxbf.bottomRows(nb1);
        auto Gb2bf = Gbxbf.topRows(nb2);
        auto Gb1bl = Gbxbl.bottomRows(nb1);
        auto Gb2bl = Gbxbl.topRows(nb2);

        auto Gbfb1 = Gbfbx.rightCols(nb1);
        auto Gbfb2 = Gbfbx.leftCols(nb2);
        auto Gblb1 = Gblbx.rightCols(nb1);
        auto Gblb2 = Gblbx.leftCols(nb2);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx.noalias() = lhs.H.diagonalRef()(ivx);

        // Calculate matrix G' = R * G * tr(R)
        G.dense = R * G.dense * tr(R);

        // The auxiliary matrix Tbxbx = Sbxn1 * Bn1bx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tb2b2 = Tbxbx.topLeftCorner(nb2, nb2);
        auto Tb2b1 = Tbxbx.topRightCorner(nb2, nb1);
        auto Tb1b2 = Tbxbx.bottomLeftCorner(nb1, nb2);
        auto Tb1b1 = Tbxbx.bottomRightCorner(nb1, nb1);

        // The auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        auto Bn1bx = mat.rightCols(nbx).middleRows(nbx, nn1);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(m + nn2, m + nn2);

        auto Mn2 = M.topRows(nn2);
        auto Mb1 = M.middleRows(nn2, nb1);
        auto Mb2 = M.middleRows(nn2 + nb1, nb2);
        auto Mbf = M.middleRows(nn2 + nb1 + nb2, nbf);
        auto Ml  = M.bottomRows(nl);

        auto Mn2n2 = Mn2.leftCols(nn2);
        auto Mn2b1 = Mn2.middleCols(nn2, nb1);
        auto Mn2b2 = Mn2.middleCols(nn2 + nb1, nb2);
        auto Mn2bf = Mn2.middleCols(nn2 + nb1 + nb2, nbf);
        auto Mn2l = Mn2.rightCols(nl);

        auto Mb1n2 = Mb1.leftCols(nn2);
        auto Mb1b1 = Mb1.middleCols(nn2, nb1);
        auto Mb1b2 = Mb1.middleCols(nn2 + nb1, nb2);
        auto Mb1bf = Mb1.middleCols(nn2 + nb1 + nb2, nbf);
        auto Mb1l = Mb1.rightCols(nl);

        auto Mb2n2 = Mb2.leftCols(nn2);
        auto Mb2b1 = Mb2.middleCols(nn2, nb1);
        auto Mb2b2 = Mb2.middleCols(nn2 + nb1, nb2);
        auto Mb2bf = Mb2.middleCols(nn2 + nb1 + nb2, nbf);
        auto Mb2l = Mb2.rightCols(nl);

        auto Mbfn2 = Mbf.leftCols(nn2);
        auto Mbfb1 = Mbf.middleCols(nn2, nb1);
        auto Mbfb2 = Mbf.middleCols(nn2 + nb1, nb2);
        auto Mbfbf = Mbf.middleCols(nn2 + nb1 + nb2, nbf);
        auto Mbfl = Mbf.rightCols(nl);

        auto Mln2 = Ml.leftCols(nn2);
        auto Mlb1 = Ml.middleCols(nn2, nb1);
        auto Mlb2 = Ml.middleCols(nn2 + nb1, nb2);
        auto Mlbf = Ml.middleCols(nn2 + nb1 + nb2, nbf);
        auto Mll = Ml.rightCols(nl);

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
        Mlb1.noalias()    = Gblb1;

        // Setting the columns of M with dimension nb2
        Mn2b2.noalias()   = -tr(Sb2n2) * diag(Hb2b2);
        Mb1b2.noalias()   = (Tb1b2 - Gb1b2)*diag(Hb2b2);
        Mb2b2.noalias()   = (Tb2b2 - Gb2b2)*diag(Hb2b2);
        Mb2b2.diagonal() += ones(nb2);
        Mbfb2.noalias()   = -Gbfb2 * diag(Hb2b2);
        Mlb2.noalias()    = -Gblb2 * diag(Hb2b2);

        // Setting the columns of M with dimension nbf
        Mn2bf.setZero();
        Mb1bf.noalias() = Gb1bf;
        Mb2bf.noalias() = Gb2bf;
        Mbfbf.noalias() = Gbfbf;
         Mlbf.noalias() = Gblbf;

        // Setting the columns of M with dimension nl
        Mn2l.setZero();
        Mb1l.noalias() = Gb1bl;
        Mb2l.noalias() = Gb2bl;
        Mbfl.noalias() = Gbfbl;
         Mll.noalias() = Gblbl;

        // Computing the LU decomposition of matrix M
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspace(SaddlePointMatrix lhs) -> void
    {
        switch(lhs.G.structure) {
            case MatrixStructure::Zero: decomposeNullspaceZeroG(lhs); break;
            default: decomposeNullspaceDenseG(lhs); break;
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspaceZeroG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a dense structure
        H.setDense(n);

        // Set the G matrix to zero structure
        G.setZero();

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // The sub-matrices in H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.dense.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnxnx = Hx.bottomRightCorner(nnx, nnx);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx << lhs.H(ivx);

        // The matrix M where we setup the coefficient matrix of the equations
        auto M = mat.topLeftCorner(nnx, nnx);

        // Calculate the coefficient matrix M of the system of linear equations
        M.noalias() = Hnxnx;
        M += tr(Sbxnx) * Hbxbx * Sbxnx;
        M -= Hnxbx * Sbxnx;
        M -= tr(Sbxnx) * Hbxnx;

        // Compute the LU decomposition of M.
        if(nnx) lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspaceDenseG(SaddlePointMatrix lhs) -> void
    {
        // Set the H matrix to a dense structure
        H.setDense(n);

        // Set the G matrix to dense structure
        G.setDense(m);

        // Set the G matrix from the given saddle point matrix
        G.dense << lhs.G;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // The sub-matrices in H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.dense.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnxnx = Hx.bottomRightCorner(nnx, nnx);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.dense.topRows(nbx);
        auto Gbf   = G.dense.middleRows(nbx, nbf);
        auto Gbl   = G.dense.bottomRows(nl);
        auto Gbxbx = Gbx.leftCols(nbx);
        auto Gbxbf = Gbx.middleCols(nbx, nbf);
        auto Gbxbl = Gbx.rightCols(nl);
        auto Gbfbx = Gbf.leftCols(nbx);
        auto Gbfbf = Gbf.middleCols(nbx, nbf);
        auto Gbfbl = Gbf.rightCols(nl);
        auto Gblbx = Gbl.leftCols(nbx);
        auto Gblbf = Gbl.middleCols(nbx, nbf);
        auto Gblbl = Gbl.rightCols(nl);

        // The indices of the free variables
        auto ivx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx << lhs.H(ivx);

        // Calculate matrix G' = R * G * tr(R)
        G.dense = R * G.dense * tr(R);

        // Auxliary matrix expressions
        const auto Ibxbx = identity(nbx, nbx);
        const auto Obfnx = zeros(nbf, nnx);
        const auto Oblnx = zeros(nl, nnx);

        // The matrix M where we setup the coefficient matrix of the equations
        auto M   = mat.topLeftCorner(nnx + m, nnx + m);
        auto Mnx = M.topRows(nnx);
        auto Mbx = M.middleRows(nnx, nbx);
        auto Mbf = M.middleRows(nnx + nbx, nbf);
        auto Mbl = M.bottomRows(nl);

        auto Mnxnx = Mnx.leftCols(nnx);
        auto Mnxbx = Mnx.middleCols(nnx, nbx);
        auto Mnxbf = Mnx.middleCols(nnx + nbx, nbf);
        auto Mnxbl = Mnx.rightCols(nl);

        auto Mbxnx = Mbx.leftCols(nnx);
        auto Mbxbx = Mbx.middleCols(nnx, nbx);
        auto Mbxbf = Mbx.middleCols(nnx + nbx, nbf);
        auto Mbxbl = Mbx.rightCols(nl);

        auto Mbfnx = Mbf.leftCols(nnx);
        auto Mbfbx = Mbf.middleCols(nnx, nbx);
        auto Mbfbf = Mbf.middleCols(nnx + nbx, nbf);
        auto Mbfbl = Mbf.rightCols(nl);

        auto Mblnx = Mbl.leftCols(nnx);
        auto Mblbx = Mbl.middleCols(nnx, nbx);
        auto Mblbf = Mbl.middleCols(nnx + nbx, nbf);
        auto Mblbl = Mbl.rightCols(nl);

        Mnxnx.noalias() = Hnxnx; Mnxnx -= Hnxbx*Sbxnx;     // Mnxnx = Hnxnx - Hnxbx*Sbxnx
        Mnxbx.noalias() = tr(Sbxnx); Mnxbx -= Hnxbx*Gbxbx; // Mnxbx = tr(Sbxnx) - Hnxbx*Gbxbx
        Mnxbf.noalias() = -Hnxbx*Gbxbf;
        Mnxbl.noalias() = -Hnxbx*Gbxbl;

        Mbxnx.noalias() = Hbxnx; Mbxnx -= Hbxbx*Sbxnx; // Mbxnx = Hbxnx - Hbxbx*Sbxnx
        Mbxbx.noalias() = Ibxbx; Mbxbx -= Hbxbx*Gbxbx; // Mbxbx = Ibxbx - Hbxbx*Gbxbx
        Mbxbf.noalias() = -Hbxbx*Gbxbf;
        Mbxbl.noalias() = -Hbxbx*Gbxbl;

        Mbfnx.noalias() = Obfnx;
        Mbfbx.noalias() = Gbfbx;
        Mbfbf.noalias() = Gbfbf;
        Mbfbl.noalias() = Gbfbl;

        Mblnx.noalias() = Oblnx;
        Mblbx.noalias() = Gblbx;
        Mblbf.noalias() = Gblbf;
        Mblbl.noalias() = Gblbl;

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Solve the saddle point problem with diagonal Hessian matrix.
    auto solve(SaddlePointVector rhs, SaddlePointSolution sol) -> Result
    {
        Result res;

        // Check if the saddle point matrix is degenerate, with no free variables.
        if(degenerate)
            solveDegenerateCase(rhs, sol);

        else switch(options.method)
        {
        case SaddlePointMethod::Nullspace: solveNullspace(rhs, sol); break;
        case SaddlePointMethod::Rangespace: solveRangespace(rhs, sol); break;
        default: solveFullspace(rhs, sol); break;
        }

        return res.stop();
    }

    /// Solve the saddle point problem for the degenerate case of no free variables.
    auto solveDegenerateCase(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        sol.x = rhs.a;

        if(G.structure == MatrixStructure::Dense)
            sol.y.noalias() = lu.solve(rhs.b);
        else
            sol.y.fill(0.0);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveFullspace(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        switch(G.structure) {
            case MatrixStructure::Zero: solveFullspaceZeroG(rhs, sol); break;
            default: solveFullspaceDenseG(rhs, sol); break;
        }
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveFullspaceZeroG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x;
        auto y = sol.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Alias to the matrices of the canonicalization process
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // View to the sub-vectors of right-hand side vector a.
        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // View to the sub-vectors of right-hand side vector b.
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);

        // Retrieve the values of a using the ordering of the free and fixed variables
        a.noalias() = rhs.a(iordering);

        // Calculate b' = R * b
        b.noalias() = R * rhs.b;

        // Calculate bbx'' = bbx' - Sbxnf*anf
        bbx -= Sbxnf * anf;

        // Calculate bbf'' = bbf' - Sbfnf*anf - abf
        bbf -= Sbfnf * anf + abf;

        // View to the right-hand side vector r of the system of linear equations
        auto r = vec.head(nx + nbx);

        // Update the vector r = [ax b]
        r << ax, bbx;

        // Solve the system of linear equations using the LU decomposition of M.
        r.noalias() = lu.solve(r);

        // Get the result of xnx from r
        ax.noalias() = r.head(nx);

        // Get the result of y' from r into bbx
        bbx.noalias() = r.tail(nbx);

        // Compute y = tr(R) * y'
        y.noalias() = tr(R)*b;

        // Permute back the variables x to their original ordering
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveFullspaceDenseG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x;
        auto y = sol.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Alias to the matrices of the canonicalization process
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // View to the sub-vectors of right-hand side vector a.
        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // View to the sub-vectors of right-hand side vector b.
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);

        // Retrieve the values of a using the ordering of the free and fixed variables
        a.noalias() = rhs.a(iordering);

        // Calculate b' = R * b
        b.noalias() = R * rhs.b;

        // Calculate bbx'' = bbx' - Sbxnf*anf
        bbx -= Sbxnf * anf;

        // Calculate bbf'' = bbf' - Sbfnf*anf - abf
        bbf -= Sbfnf * anf + abf;

        // View to the right-hand side vector r of the system of linear equations
        auto r = vec.head(nx + m);

        // Update the vector r = [ax b]
        r << ax, b;

        // Solve the system of linear equations using the LU decomposition of M.
        r.noalias() = lu.solve(r);

        // Get the result of xnx from r
        ax.noalias() = r.head(nx);

        // The y' vector as the tail of the solution of the linear system
        auto yp = r.tail(m);

        // Compute y = tr(R) * y'
        y.noalias() = tr(R)*yp;

        // Permute back the variables x to their original ordering
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceAux(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        switch(G.structure) {
            case MatrixStructure::Zero: solveRangespaceZeroG(rhs, sol); break;
            default: solveRangespaceDenseG(rhs, sol); break;
        }
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespace(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        switch(H.structure) {
            case MatrixStructure::Dense: solveNullspace(rhs, sol); break;
            default: solveRangespaceAux(rhs, sol); break;
        }
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceZeroG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x;
        auto y = sol.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sb1n1 = Sbxnx.bottomRightCorner(nb1, nn1);
        auto Sb2n1 = Sbxnx.topRightCorner(nb2, nn1);
        auto Sb2nx = Sbxnx.topRows(nb2);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = H.diagonal.head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);

        auto ax  = a.head(nx);
        auto af  = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto anf = af.tail(nnf);
        auto ab1 = abx.tail(nb1);
        auto ab2 = abx.head(nb2);
        auto an1 = anx.tail(nn1);
        auto an2 = anx.head(nn2);

        auto bbx = b.head(nbx);
        auto bb1 = bbx.tail(nb1);
        auto bb2 = bbx.head(nb2);

        a.noalias() = rhs.a(iordering);

        b.noalias() = R * rhs.b;

        anx -= tr(Sb2nx) * ab2;
        bbx -= Sbxnf * anf;

        an1.noalias() = an1/Hn1n1;

        bb1 -= ab1/Hb1b1;
        bb1 -= Sb1n1 * an1;

        bb2 -= Sb2n1 * an1;

        auto r = vec.head(nb1 + nb2 + nn2);

        auto xn2 = r.head(nn2);
        auto yb1 = r.segment(nn2, nb1);
        auto xb2 = r.segment(nn2 + nb1, nb2);

        r << an2, bb1, bb2;

        r.noalias() = lu.solve(r);

        ab1.noalias() = (ab1 - yb1)/Hb1b1;
        bb2.noalias() = (ab2 - Hb2b2 % xb2);
        an1.noalias() -= (tr(Sb1n1)*yb1 + tr(Sb2n1)*(bb2 - ab2))/Hn1n1;

        an2.noalias() = xn2;
        bb1.noalias() = yb1;
        ab2.noalias() = xb2;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R) * b;

        // Permute back the variables `x` to their original ordering
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDenseG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = sol.x;
        auto y = sol.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);
        auto Sb1n1 = Sbxnx.bottomRightCorner(nb1, nn1);
        auto Sb2n1 = Sbxnx.topRightCorner(nb2, nn1);
        auto Sb2nx = Sbxnx.topRows(nb2);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hx    = H.diagonal.head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.tail(nb1);
        auto Hb2b2 = Hbxbx.head(nb2);
        auto Hn1n1 = Hnxnx.tail(nn1);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.dense.topRows(nbx);
        auto Gbf   = G.dense.middleRows(nbx, nbf);
        auto Gbl   = G.dense.bottomRows(nl);
        auto Gbxbx = Gbx.leftCols(nbx);
        auto Gbfbx = Gbf.leftCols(nbx);
        auto Gblbx = Gbl.leftCols(nbx);

        auto Gb1b2 = Gbxbx.bottomLeftCorner(nb1, nb2);
        auto Gb2b2 = Gbxbx.topLeftCorner(nb2, nb2);

        auto Gbfb2 = Gbfbx.leftCols(nb2);
        auto Gblb2 = Gblbx.leftCols(nb2);

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

        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);
        auto bl  = b.tail(nl);
        auto bb1 = bbx.tail(nb1);
        auto bb2 = bbx.head(nb2);

        a.noalias() = rhs.a(iordering);

        b.noalias() = R * rhs.b;

        anx -= tr(Sb2nx) * ab2;
        bbx -= Sbxnf * anf;
        bbf -= Sbfnf * anf + abf;

        an1.noalias() = an1/Hn1n1;

        bb1 -= ab1/Hb1b1;
        bb1 -= Sb1n1 * an1;
        bb1 -= Gb1b2 * ab2;

        bb2 -= Sb2n1 * an1;
        bb2 -= Gb2b2 * ab2;
        bbf -= Gbfb2 * ab2;
        bl  -= Gblb2 * ab2;

        auto r = vec.head(nn2 + m);

        auto xn2 = r.head(nn2);
        auto yb1 = r.segment(nn2, nb1);
        auto xb2 = r.segment(nn2 + nb1, nb2);
        auto ybf = r.segment(nn2 + nb1 + nb2, nbf);
        auto yl = r.tail(nl);

        r << an2, bb1, bb2, bbf, bl;

        r.noalias() = lu.solve(r);

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
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspace(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        switch(G.structure) {
            case MatrixStructure::Zero: solveNullspaceZeroG(rhs, sol); break;
            default: solveNullspaceDenseG(rhs, sol); break;
        }
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspaceZeroG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-vectors of right-hand side vector a = [ax af]
        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // Views to the sub-vectors of right-hand side vector b = [bx bf bl]
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // Views to the sub-matrices of H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.dense.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);

        // The vector y' = [ybx' ybf' ybl']
        auto yp   = vec.head(m);
        auto ypbx = yp.head(nbx);
        auto ypbf = yp.segment(nbx, nbf);
        auto ypbl = yp.tail(nl);

        // Set vectors `ax` and `af` using values from `a`
        a.noalias() = rhs.a(iordering);

        // Calculate b' = R*b
        b.noalias() = R*rhs.b;

        // Calculate bbx'' = bbx' - Sbxnf*anf
        bbx -= Sbxnf*anf;

        // Calculate bbf'' = bbf' - abf - Sbfnf*anf
        bbf -= abf + Sbfnf*anf;

        // Set ypx' = abx
        ypbx = abx;

        // Calculate abx' = abx - Hbxbx*bbx''
        abx -= Hbxbx*bbx;

        // Calculate anx' = anx - Hnxbx*bbx'' - tr(Sbxnx)*abx'
        anx -= Hnxbx*bbx + tr(Sbxnx)*abx;

        // Solve the system of linear equations
        if(nnx) anx.noalias() = lu.solve(anx);

        // Alias to solution vectors x and y
        auto x = sol.x;
        auto y = sol.y;

        // Calculate xbx and store in abx
        abx.noalias() = bbx - Sbxnx*anx;

        // Calculate y'
        ypbx -= Hbxbx*abx + Hbxnx*anx;
        ypbf.noalias() = zeros(nbf);
        ypbl.noalias() = zeros(nl);

        // Calculate y = tr(R) * y'
        y.noalias() = tr(R) * yp;

        // Set back the values of x currently stored in a
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspaceDenseG(SaddlePointVector rhs, SaddlePointSolution sol) -> void
    {
        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-vectors of right-hand side vector a = [ax af]
        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);

        // Views to the sub-vectors of right-hand side vector b = [bx bf bl]
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);
        auto bbl = b.tail(nl);

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // Views to the sub-matrices of H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.dense.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);

        // The sub-matrix Gbx in G' = R * G * tr(R)
        auto Gbx = G.dense.topRows(nbx);

        // The indices of the free and fixed variables
        auto ivx = iordering.head(nx);
        auto ivf = iordering.tail(nf);

        // The right-hand side vector r = [rnx rbx rbf rbl]
        auto r = vec.head(nnx + m);
        auto rnx = r.head(nnx);
        auto rb  = r.tail(m);
        auto rbx = rb.head(nbx);
        auto rbf = rb.segment(nbx, nbf);
        auto rbl = rb.tail(nl);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = rhs.a(ivx);
        af.noalias() = rhs.a(ivf);

        // Calculate b' = R*b
        b.noalias() = R*rhs.b;

        // Calculate bbx'' = bbx' - Sbxnf*anf
        bbx -= Sbxnf*anf;

        // Calculate bbf'' = bbf' - abf - Sbfnf*anf
        bbf -= abf + Sbfnf*anf;

        // Set right-hand side vector r = [rnx rbx rbf rl]
        rnx.noalias() = anx - Hnxbx*bbx;
        rbx.noalias() = abx - Hbxbx*bbx;
        rbf.noalias() = bbf;
        rbl.noalias() = bbl;

        // Solve the system of linear equations
        r.noalias() = lu.solve(r);

        // Alias to solution vectors x and y
        auto x = sol.x;
        auto y = sol.y;

        // Calculate y = tr(R) * y'
        y.noalias() = tr(R) * rb;

        // Calculate x = [xbx xnx] using auxiliary abx and anx
        anx.noalias() = rnx;
        abx.noalias() = bbx - Sbxnx*anx - Gbx*rb;

        // Set back the values of x currently stored in a
        x(iordering).noalias() = a;
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

auto SaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->options;
}

auto SaddlePointSolver::reorderVariables(VectorXiConstRef ordering) -> void
{
    pimpl->reorderVariables(ordering);
}

auto SaddlePointSolver::initialize(MatrixConstRef A) -> Result
{
    return pimpl->initialize(A);
}

auto SaddlePointSolver::decompose(SaddlePointMatrix lhs) -> Result
{
    return pimpl->decompose(lhs);
}

auto SaddlePointSolver::solve(SaddlePointVector rhs, SaddlePointSolution sol) -> Result
{
    return pimpl->solve(rhs, sol);
}

} // namespace Optima

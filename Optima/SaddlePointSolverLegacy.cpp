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

#include "SaddlePointSolverLegacy.hpp"

// C++ includes
#include <cassert>
#include <cmath>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/LU>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/ExtendedCanonicalizer.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverLegacy::Impl
{
    ExtendedCanonicalizer canonicalizer; ///< The canonicalizer of the Jacobian matrix *W = [A; J]*.

    SaddlePointOptions options; ///< The options used to solve the saddle point problems.

    Index n   = 0; ///< The number of columns in the Jacobian matrix *W = [A; J]*
    Index m   = 0; ///< The number of rows in the Jacobian matrix *W = [A; J]*
    Index ml  = 0; ///< The number of rows in matrix *A*.
    Index mn  = 0; ///< The number of rows in matrix *J*.
    Index nb  = 0; ///< The number of basic variables.
    Index nn  = 0; ///< The number of non-basic variables.
    Index nl  = 0; ///< The number of linearly dependent rows in *W = [A; J]*
    Index nx  = 0; ///< The number of *free* variables.
    Index nf  = 0; ///< The number of *fixed* variables.
    Index nbx = 0; ///< The number of *free basic* variables.
    Index nbf = 0; ///< The number of *fixed basic* variables.
    Index nnx = 0; ///< The number of *free non-basic* variables.
    Index nnf = 0; ///< The number of *fixed non-basic* variables.
    Index nbe = 0; ///< The number of *pivot free basic* variables.
    Index nne = 0; ///< The number of *pivot free non-basic* variables.
    Index nbi = 0; ///< The number of *non-pivot free basic* variables.
    Index nni = 0; ///< The number of *non-pivot free non-basic* variables.

    Vector weights; ///< The priority weights for the selection of basic variables.

    Matrix W; ///< The 'W = [A; J]' matrix in the saddle point matrix.
    Matrix H; ///< The 'H' matrix in the saddle point matrix.
    Matrix G; ///< The 'G' matrix in the saddle point matrix.

    Vector a; ///< The workspace for the right-hand side vectors a and b
    Vector b; ///< The workspace for the right-hand side vectors a and b

    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods.

    Indices iordering; ///< The ordering of the variables as (free-basic, free-non-basic, fixed-basic, fixed-non-basic)

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

    bool degenerate = false;  ///< The boolean flag that indicates that the decomposed saddle point matrix was degenerate with no free variables.

    /// Construct a default SaddlePointSolverLegacy::Impl instance.
    Impl()
    {}

    /// Construct a SaddlePointSolverLegacy::Impl instance with given data.
    Impl(SaddlePointSolverLegacyInitArgs args)
    : n(args.n), m(args.m), canonicalizer(args.A)
    {
        // Ensure consistent and proper dimensions
        assert(args.n > 0);
        assert(args.A.rows() == 0 || args.A.rows() <= m);
        assert(args.A.rows() == 0 || args.A.cols() == n);

        // Set the number of rows in matrices *A* and *J*
        ml = args.A.rows();
        mn = m - ml;

        // Allocate auxiliary memory
        a.resize(n);
        b.resize(m);
        mat.resize(n + m, n + m);
        vec.resize(n + m);
        weights.resize(n);
        W.resize(m, n);
        H.resize(n, n);

        // Initialize the upper part of W = [A; J]
        W.topRows(ml) = args.A;

        // Initialize the initial ordering of the variables
        iordering = indices(n);
    }

    /// Canonicalize the *W = [A; J]* matrix of the saddle point problem.
    auto canonicalize(SaddlePointSolverLegacyCanonicalizeArgs args) -> void
    {
        // Ensure number of variables is positive.
        assert(n > 0);

        // Update the lower part of W = [A; J]
        W.bottomRows(mn) = args.J;

        // Update the number of fixed and free variables
        nf = args.ifixed.size();
        nx = n - nf;

        // Determine if the saddle point matrix is degenerate
        degenerate = nx == 0;

        // Skip the rest if there is no free variables
        if(degenerate)
            return;

        // The ordering of the variables as (free variables, fixed variables)
        moveIntersectionRight(iordering, args.ifixed);

        // The indices of the fixed (ifixed) variables
        const auto ifixed = iordering.tail(nf);

        // Update the priority weights for the update of the canonical form
        const auto fakezero = std::numeric_limits<double>::min(); // 2.22507e-308
        weights.noalias() = args.H.diagonal();
        weights.noalias() = (weights.array() == 0).select(fakezero, weights); // replace zero by fakezero to avoid division by zero
        weights.noalias() = abs(inv(weights));

        // Set negative priority weights for the fixed variables
        weights(ifixed).noalias() = -linspace(nf, 1, nf);

        // Update the canonical form and the ordering of the variables
        canonicalizer.updateWithPriorityWeights(args.J, weights);

        // Update the number of basic and non-basic variables
        nb = canonicalizer.numBasicVariables();
        nn = canonicalizer.numNonBasicVariables();

        // Update the number of linearly dependent rows in *W = [A; J]*
        nl = m - nb;

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = canonicalizer.indicesBasicVariables();
        const auto inonbasic = canonicalizer.indicesNonBasicVariables();

        // Get the S matrix of the canonical form of *W = [A; J]*
        const auto S = canonicalizer.S();

        // Find the number of fixed basic variables (those with weights below zero)
        // Walk from last to first basic variable, since the fixed basic variables are at the end of ibasic
        nbf = 0; while(nbf < nb && weights[ibasic[nb - nbf - 1]] < 0.0) ++nbf;

        // Find the number of fixed non-basic variables (those with weights below zero)
        // Walk from last to first non-basic variable, since the fixed non-basic variables are at the end of inonbasic
        nnf = 0; while(nnf < nn && weights[inonbasic[nn - nnf - 1]] < 0.0) ++nnf;

        // Update the number of free basic and free non-basic variables
        nbx = nb - nbf;
        nnx = nn - nnf;

        // Update the number of non-pivot free basic variables.
        // Walk from first to last basic variable, since the free basic variables are at the beginning of ibasic
        nbi = 0; while(nbi < nbx && weights[ibasic[nbi]] > 1.0) ++nbi;

        // Update the number of non-pivot free non-basic variables.
        // Walk from first to last non-basic variable, since the free non-basic variables are at the beginning of inonbasic
        nni = 0; while(nni < nnx && weights[inonbasic[nni]] > norminf(S.col(nni))) ++nni;

        // Update the number of pivot free basic and non-basic variables.
        nbe = nbx - nbi;
        nne = nnx - nni;

        // Update the ordering of the free variables as xx = [xbx xnx]
        iordering.head(nx).head(nbx) = ibasic.head(nbx);
        iordering.head(nx).tail(nnx) = inonbasic.head(nnx);

        // Update the ordering of the fixed variables as xf = [xbf xnf]
        iordering.tail(nf).head(nbf) = ibasic.tail(nbf);
        iordering.tail(nf).tail(nnf) = inonbasic.tail(nnf);
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Ensure number of variables is positive.
        assert(n > 0);

        // Check if the saddle point matrix is degenerate, with no free variables.
        if(degenerate)
            decomposeDegenerateCase(args);

        else switch(options.method)
        {
        case SaddlePointMethod::Nullspace: decomposeNullspace(args); break;
        case SaddlePointMethod::Rangespace: decomposeRangespace(args); break;
        default: decomposeFullspace(args); break;
        }
    }

    /// Decompose the saddle point matrix for the degenerate case of no free variables.
    auto decomposeDegenerateCase(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        if(isDenseMatrix(args.G))
        {
            // Set the G matrix to dense structure
            G.resize(m, m);

            // Set the G matrix from the given saddle point matrix
            G = args.G;

            // Compute the LU decomposition of G.
            lu.compute(G);
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspace(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        if(isZeroMatrix(args.G)) decomposeFullspaceZeroG(args);
        else decomposeFullspaceDenseG(args);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspaceZeroG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the G matrix to an empty matrix
        G = {};

        // The indices of the free variables
        auto jx = iordering.head(nx);

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

        // Set the H + D block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = args.H(jx, jx);

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a LU decomposition method.
    auto decomposeFullspaceDenseG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the H matrix to a dense structure
        H.resize(n, n);

        // Set the G matrix to dense structure
        G.resize(m, m);

        // Set the G matrix from the given saddle point matrix
        G = args.G;

        // The indices of the free variables
        auto jx = iordering.head(nx);

        // Alias to the matrices of the canonicalization process
        auto Sbn   = canonicalizer.S();
        auto Sbxnx = Sbn.topLeftCorner(nbx, nnx);

        auto R = canonicalizer.R();

        auto Ibxbx = identity(nbx, nbx);

        // Initialize workspace with zeros
        mat.fill(0.0);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(m + nx, m + nx);

        // Set the H + D block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = args.H(jx, jx);

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.middleCols(nx, nbx).topRows(nx) << Ibxbx, tr(Sbxnx);
        M.middleRows(nx, nbx).leftCols(nx) << Ibxbx, Sbxnx;

        // Set the zero blocks of M on the top-right and bottom-left corners
        M.topRightCorner(nx, nbf + nl).setZero();
        M.bottomLeftCorner(nbf + nl, nx).setZero();

        // Set the G block of M on the bottom-right corner
        M.bottomRightCorner(m, m) = R * G * tr(R);

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespace(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        if(isZeroMatrix(args.G)) decomposeRangespaceZeroG(args);
        else decomposeRangespaceDenseG(args);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceZeroG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the G matrix to an empty matrix
        G = {};

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbeni = Sbxnx.bottomLeftCorner(nbe, nni);
        auto Sbini = Sbxnx.topLeftCorner(nbi, nni);
        auto Sbxne = Sbxnx.rightCols(nne);

        // The diagonal entries in H of the free variables, with Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hx    = H.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.tail(nbe);
        auto Hbibi = Hbxbx.head(nbi);
        auto Hnene = Hnxnx.tail(nne);
        auto Hnini = Hnxnx.head(nni);

        // The indices of the free variables
        auto jx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx.noalias() = args.H.diagonal()(jx);

        // The auxiliary matrix Tbxbx = Sbxne * Bnebx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tbibi = Tbxbx.topLeftCorner(nbi, nbi);
        auto Tbibe = Tbxbx.topRightCorner(nbi, nbe);
        auto Tbebi = Tbxbx.bottomLeftCorner(nbe, nbi);
        auto Tbebe = Tbxbx.bottomRightCorner(nbe, nbe);

        // The auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        auto Bnebx = mat.rightCols(nbx).middleRows(nbx, nne);

        // Initialize workspace with zeros
        mat.fill(0.0);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(nbe + nbi + nni, nbe + nbi + nni);

        auto Mni = M.topRows(nni);
        auto Mbe = M.middleRows(nni, nbe);
        auto Mbi = M.middleRows(nni + nbe, nbi);

        auto Mnini = Mni.leftCols(nni);
        auto Mnibe = Mni.middleCols(nni, nbe);
        auto Mnibi = Mni.middleCols(nni + nbe, nbi);

        auto Mbeni = Mbe.leftCols(nni);
        auto Mbebe = Mbe.middleCols(nni, nbe);
        auto Mbebi = Mbe.middleCols(nni + nbe, nbi);

        auto Mbini = Mbi.leftCols(nni);
        auto Mbibe = Mbi.middleCols(nni, nbe);
        auto Mbibi = Mbi.middleCols(nni + nbe, nbi);

        // Computing the auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        Bnebx.noalias() = diag(inv(Hnene)) * tr(Sbxne);

        // Computing the auxiliary matrix Tbxbx = Sbxne * Bnebx
        Tbxbx.noalias() = Sbxne * Bnebx;

        // Setting the columns of M with dimension nni
        Mnini           = diag(Hnini);
        Mbeni.noalias() = Sbeni;
        Mbini.noalias() = Sbini;

        // Setting the columns of M with dimension nbe
        Mnibe.noalias()   = tr(Sbeni);
        Mbebe.noalias()   = -Tbebe;
        Mbebe.diagonal() -= inv(Hbebe);
        Mbibe.noalias()   = -Tbibe;

        // Setting the columns of M with dimension nbi
        Mnibi.noalias()   = -tr(Sbini) * diag(Hbibi);
        Mbebi.noalias()   = Tbebi*diag(Hbibi);
        Mbibi.noalias()   = Tbibi*diag(Hbibi);
        Mbibi.diagonal() += ones(nbi);

        // Computing the LU decomposition of matrix M
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a rangespace diagonal method.
    auto decomposeRangespaceDenseG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the G matrix from the given saddle point matrix
        G = args.G;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbeni = Sbxnx.bottomLeftCorner(nbe, nni);
        auto Sbini = Sbxnx.topLeftCorner(nbi, nni);
        auto Sbxne = Sbxnx.rightCols(nne);

        // The diagonal entries in H of the free variables, with Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hx    = H.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.tail(nbe);
        auto Hbibi = Hbxbx.head(nbi);
        auto Hnene = Hnxnx.tail(nne);
        auto Hnini = Hnxnx.head(nni);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.topRows(nbx);
        auto Gbf   = G.middleRows(nbx, nbf);
        auto Gbl   = G.bottomRows(nl);
        auto Gbxbx = Gbx.leftCols(nbx);
        auto Gbxbf = Gbx.middleCols(nbx, nbf);
        auto Gbxbl = Gbx.rightCols(nl);
        auto Gbfbx = Gbf.leftCols(nbx);
        auto Gbfbf = Gbf.middleCols(nbx, nbf);
        auto Gbfbl = Gbf.rightCols(nl);
        auto Gblbx = Gbl.leftCols(nbx);
        auto Gblbf = Gbl.middleCols(nbx, nbf);
        auto Gblbl = Gbl.rightCols(nl);

        auto Gbebe = Gbxbx.bottomRightCorner(nbe, nbe);
        auto Gbebi = Gbxbx.bottomLeftCorner(nbe, nbi);
        auto Gbibe = Gbxbx.topRightCorner(nbi, nbe);
        auto Gbibi = Gbxbx.topLeftCorner(nbi, nbi);

        auto Gbebf = Gbxbf.bottomRows(nbe);
        auto Gbibf = Gbxbf.topRows(nbi);
        auto Gbebl = Gbxbl.bottomRows(nbe);
        auto Gbibl = Gbxbl.topRows(nbi);

        auto Gbfbe = Gbfbx.rightCols(nbe);
        auto Gbfbi = Gbfbx.leftCols(nbi);
        auto Gblbe = Gblbx.rightCols(nbe);
        auto Gblbi = Gblbx.leftCols(nbi);

        // The indices of the free variables
        auto jx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx.noalias() = args.H.diagonal()(jx);

        // Calculate matrix G' = R * G * tr(R)
        G = R * G * tr(R);  // TODO: Try R * args.G * tr(R)

        // The auxiliary matrix Tbxbx = Sbxne * Bnebx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tbibi = Tbxbx.topLeftCorner(nbi, nbi);
        auto Tbibe = Tbxbx.topRightCorner(nbi, nbe);
        auto Tbebi = Tbxbx.bottomLeftCorner(nbe, nbi);
        auto Tbebe = Tbxbx.bottomRightCorner(nbe, nbe);

        // The auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        auto Bnebx = mat.rightCols(nbx).middleRows(nbx, nne);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(m + nni, m + nni);

        auto Mni = M.topRows(nni);
        auto Mbe = M.middleRows(nni, nbe);
        auto Mbi = M.middleRows(nni + nbe, nbi);
        auto Mbf = M.middleRows(nni + nbe + nbi, nbf);
        auto Ml  = M.bottomRows(nl);

        auto Mnini = Mni.leftCols(nni);
        auto Mnibe = Mni.middleCols(nni, nbe);
        auto Mnibi = Mni.middleCols(nni + nbe, nbi);
        auto Mnibf = Mni.middleCols(nni + nbe + nbi, nbf);
        auto Mnil = Mni.rightCols(nl);

        auto Mbeni = Mbe.leftCols(nni);
        auto Mbebe = Mbe.middleCols(nni, nbe);
        auto Mbebi = Mbe.middleCols(nni + nbe, nbi);
        auto Mbebf = Mbe.middleCols(nni + nbe + nbi, nbf);
        auto Mbel = Mbe.rightCols(nl);

        auto Mbini = Mbi.leftCols(nni);
        auto Mbibe = Mbi.middleCols(nni, nbe);
        auto Mbibi = Mbi.middleCols(nni + nbe, nbi);
        auto Mbibf = Mbi.middleCols(nni + nbe + nbi, nbf);
        auto Mbil = Mbi.rightCols(nl);

        auto Mbfni = Mbf.leftCols(nni);
        auto Mbfbe = Mbf.middleCols(nni, nbe);
        auto Mbfbi = Mbf.middleCols(nni + nbe, nbi);
        auto Mbfbf = Mbf.middleCols(nni + nbe + nbi, nbf);
        auto Mbfl = Mbf.rightCols(nl);

        auto Mlni = Ml.leftCols(nni);
        auto Mlbe = Ml.middleCols(nni, nbe);
        auto Mlbi = Ml.middleCols(nni + nbe, nbi);
        auto Mlbf = Ml.middleCols(nni + nbe + nbi, nbf);
        auto Mll = Ml.rightCols(nl);

        // Computing the auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        Bnebx.noalias() = diag(inv(Hnene)) * tr(Sbxne);

        // Computing the auxiliary matrix Tbxbx = Sbxne * Bnebx
        Tbxbx.noalias() = Sbxne * Bnebx;

        // Setting the columns of M with dimension nni
        Mnini           = diag(Hnini);
        Mbeni.noalias() = Sbeni;
        Mbini.noalias() = Sbini;
        Mbfni.setZero();
        Mlni.setZero();

        // Setting the columns of M with dimension nbe
        Mnibe.noalias()   = tr(Sbeni);
        Mbebe.noalias()   = Gbebe - Tbebe;
        Mbebe.diagonal() -= inv(Hbebe);
        Mbibe.noalias()   = Gbibe - Tbibe;
        Mbfbe.noalias()   = Gbfbe;
        Mlbe.noalias()    = Gblbe;

        // Setting the columns of M with dimension nbi
        Mnibi.noalias()   = -tr(Sbini) * diag(Hbibi);
        Mbebi.noalias()   = (Tbebi - Gbebi)*diag(Hbibi);
        Mbibi.noalias()   = (Tbibi - Gbibi)*diag(Hbibi);
        Mbibi.diagonal() += ones(nbi);
        Mbfbi.noalias()   = -Gbfbi * diag(Hbibi);
        Mlbi.noalias()    = -Gblbi * diag(Hbibi);

        // Setting the columns of M with dimension nbf
        Mnibf.setZero();
        Mbebf.noalias() = Gbebf;
        Mbibf.noalias() = Gbibf;
        Mbfbf.noalias() = Gbfbf;
         Mlbf.noalias() = Gblbf;

        // Setting the columns of M with dimension nl
        Mnil.setZero();
        Mbel.noalias() = Gbebl;
        Mbil.noalias() = Gbibl;
        Mbfl.noalias() = Gbfbl;
         Mll.noalias() = Gblbl;

        // Computing the LU decomposition of matrix M
        lu.compute(M);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspace(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        if(isZeroMatrix(args.G)) decomposeNullspaceZeroG(args);
        else decomposeNullspaceDenseG(args);
    }

    /// Decompose the coefficient matrix of the saddle point problem using a nullspace method.
    auto decomposeNullspaceZeroG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the G matrix to an empty matrix
        G = {};

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // The sub-matrices in H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnxnx = Hx.bottomRightCorner(nnx, nnx);

        // The indices of the free variables
        auto jx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx = args.H(jx, jx);

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
    auto decomposeNullspaceDenseG(SaddlePointSolverLegacyDecomposeArgs args) -> void
    {
        // Set the G matrix from the given saddle point matrix
        G = args.G;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // The sub-matrices in H, with Hx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hx    = H.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);
        auto Hnxnx = Hx.bottomRightCorner(nnx, nnx);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.topRows(nbx);
        auto Gbf   = G.middleRows(nbx, nbf);
        auto Gbl   = G.bottomRows(nl);
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
        auto jx = iordering.head(nx);

        // Retrieve the entries in H corresponding to free variables.
        Hx = args.H(jx, jx);

        // Calculate matrix G' = R * G * tr(R)
        G = R * G * tr(R); // TODO: Try G = R * args.G * tr(R)

        // Auxliary matrix expressions
        const auto Ibxbx = identity(nbx, nbx);
        const auto Obfnx = zeros(nbf, nnx);
        const auto Oblnx = zeros(nl, nnx);

        // Initialize workspace with zeros
        mat.fill(0.0);

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

    /// Solve the saddle point problem.
    auto solve(SaddlePointSolverLegacySolveArgs args) -> void
    {
        // Check if the saddle point matrix is degenerate, with no free variables.
        if(degenerate)
            solveDegenerateCase(args);

        else switch(options.method)
        {
        case SaddlePointMethod::Nullspace: solveNullspace(args); break;
        case SaddlePointMethod::Rangespace: solveRangespace(args); break;
        default: solveFullspace(args); break;
        }
    }

    /// Solve the saddle point problem.
    auto solve(SaddlePointSolverLegacySolveAdvancedArgs args) -> void
    {
        // Auxiliary references
        const auto jx = iordering.head(nx);
        const auto jf = iordering.tail(nf);

        // Use this->vec.head(n) as work space for x' where x'(free) = x(free) and x'(fixed) = 0
        auto xprime = vec.head(n);

        xprime(jx) = args.x(jx);
        xprime(jf).fill(0.0);

        // Use args.xbar as work space for a = H*x - g
        auto a = args.xbar;

        // Compute H*x', considering only diag(H) in case of rangespace method!
        // The use of x' instead of x is because H contribution from fixed
        // variables should be ignored.
        if(options.method == SaddlePointMethod::Rangespace)
            a = args.H.diagonal().cwiseProduct(xprime);
        else a = args.H * xprime;

        // Complete the calculation of H*x' - g
        a -= args.g;

        // Use args.ybar as work space for b' = [b, J*x - h]
        auto b = args.ybar;

        b.head(ml) = args.b;
        b.tail(mn) = args.J * args.x - args.h;

        // Ensure the computed xbar satisfy xbar(jf) = x(jf)!
        a(jf) = args.x(jf);

        // Compute the solution vectors x and y in the saddle point problem
        solve({ a, b, args.xbar, args.ybar });
    }

    /// Solve the saddle point problem for the degenerate case of no free variables.
    auto solveDegenerateCase(SaddlePointSolverLegacySolveArgs args) -> void
    {
        args.x = args.a;

        if(isZeroMatrix(G)) args.y.fill(0.0);
        else args.y.noalias() = lu.solve(args.b);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveFullspace(SaddlePointSolverLegacySolveArgs args) -> void
    {
        if(isZeroMatrix(G)) solveFullspaceZeroG(args);
        else solveFullspaceDenseG(args);
    }

    /// Solve the saddle point problem using a LU decomposition method.
    auto solveFullspaceZeroG(SaddlePointSolverLegacySolveArgs args) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = args.x;
        auto y = args.y;

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
        a.noalias() = args.a(iordering);

        // Calculate b' = R * b
        b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

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
    auto solveFullspaceDenseG(SaddlePointSolverLegacySolveArgs args) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = args.x;
        auto y = args.y;

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
        a.noalias() = args.a(iordering);

        // Calculate b' = R * b
        b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

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
    auto solveRangespace(SaddlePointSolverLegacySolveArgs args) -> void
    {
        if(isZeroMatrix(G)) solveRangespaceZeroG(args);
        else solveRangespaceDenseG(args);
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceZeroG(SaddlePointSolverLegacySolveArgs args) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = args.x;
        auto y = args.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbene = Sbxnx.bottomRightCorner(nbe, nne);
        auto Sbine = Sbxnx.topRightCorner(nbi, nne);
        auto Sbinx = Sbxnx.topRows(nbi);

        // The diagonal entries in H of the free variables, with Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hx    = H.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.tail(nbe);
        auto Hbibi = Hbxbx.head(nbi);
        auto Hnene = Hnxnx.tail(nne);

        auto ax  = a.head(nx);
        auto af  = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto anf = af.tail(nnf);
        auto abe = abx.tail(nbe);
        auto abi = abx.head(nbi);
        auto ane = anx.tail(nne);
        auto ani = anx.head(nni);

        auto bbx = b.head(nbx);
        auto bbe = bbx.tail(nbe);
        auto bbi = bbx.head(nbi);

        a.noalias() = args.a(iordering);

        auto jf = iordering.tail(nf);

        // Alias to matrix A in W = [A; J]
        const auto Wf = W(Eigen::all, jf);

        b.noalias() = R * (args.b - Wf*af);
        // b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

        anx -= tr(Sbinx) * abi;
        // bbx -= Sbxnf * anf;

        ane.noalias() = ane/Hnene;

        bbe -= abe/Hbebe;
        bbe -= Sbene * ane;

        bbi -= Sbine * ane;

        auto r = vec.head(nbe + nbi + nni);

        auto xni = r.head(nni);
        auto ybe = r.segment(nni, nbe);
        auto xbi = r.segment(nni + nbe, nbi);

        r << ani, bbe, bbi;

        r.noalias() = lu.solve(r);

        abe.noalias() = (abe - ybe)/Hbebe;
        bbi.noalias() = (abi - Hbibi % xbi);
        ane.noalias() -= (tr(Sbene)*ybe + tr(Sbine)*(bbi - abi))/Hnene;

        ani.noalias() = xni;
        bbe.noalias() = ybe;
        abi.noalias() = xbi;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R) * b;

        // Permute back the variables `x` to their original ordering
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a rangespace diagonal method.
    auto solveRangespaceDenseG(SaddlePointSolverLegacySolveArgs args) -> void
    {
        // Alias to members of the saddle point vector solution.
        auto x = args.x;
        auto y = args.y;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);
        auto Sbene = Sbxnx.bottomRightCorner(nbe, nne);
        auto Sbine = Sbxnx.topRightCorner(nbi, nne);
        auto Sbinx = Sbxnx.topRows(nbi);

        // The diagonal entries in H of the free variables, with Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hx    = H.col(0).head(nx);
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.tail(nbe);
        auto Hbibi = Hbxbx.head(nbi);
        auto Hnene = Hnxnx.tail(nne);

        // The sub-matrices in G' = R * G * tr(R)
        auto Gbx   = G.topRows(nbx);
        auto Gbf   = G.middleRows(nbx, nbf);
        auto Gbl   = G.bottomRows(nl);
        auto Gbxbx = Gbx.leftCols(nbx);
        auto Gbfbx = Gbf.leftCols(nbx);
        auto Gblbx = Gbl.leftCols(nbx);

        auto Gbebi = Gbxbx.bottomLeftCorner(nbe, nbi);
        auto Gbibi = Gbxbx.topLeftCorner(nbi, nbi);

        auto Gbfbi = Gbfbx.leftCols(nbi);
        auto Gblbi = Gblbx.leftCols(nbi);

        auto ax  = a.head(nx);
        auto af  = a.tail(nf);
        auto abx = ax.head(nbx);
        auto anx = ax.tail(nnx);
        auto abf = af.head(nbf);
        auto anf = af.tail(nnf);
        auto abe = abx.tail(nbe);
        auto abi = abx.head(nbi);
        auto ane = anx.tail(nne);
        auto ani = anx.head(nni);

        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);
        auto bl  = b.tail(nl);
        auto bbe = bbx.tail(nbe);
        auto bbi = bbx.head(nbi);

        a.noalias() = args.a(iordering);

        b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

        anx -= tr(Sbinx) * abi;
        bbx -= Sbxnf * anf;
        bbf -= Sbfnf * anf + abf;

        ane.noalias() = ane/Hnene;

        bbe -= abe/Hbebe;
        bbe -= Sbene * ane;
        bbe -= Gbebi * abi;

        bbi -= Sbine * ane;
        bbi -= Gbibi * abi;
        bbf -= Gbfbi * abi;
        bl  -= Gblbi * abi;

        auto r = vec.head(nni + m);

        auto xni = r.head(nni);
        auto ybe = r.segment(nni, nbe);
        auto xbi = r.segment(nni + nbe, nbi);
        auto ybf = r.segment(nni + nbe + nbi, nbf);
        auto yl = r.tail(nl);

        r << ani, bbe, bbi, bbf, bl;

        r.noalias() = lu.solve(r);

        abe.noalias() = (abe - ybe)/Hbebe;
        bbi.noalias() = (abi - Hbibi % xbi);
        ane.noalias() -= (tr(Sbene)*ybe + tr(Sbine)*(bbi - abi))/Hnene;

        ani.noalias() = xni;
        bbe.noalias() = ybe;
        abi.noalias() = xbi;
        bbf.noalias() = ybf;
         bl.noalias() = yl;

        // Compute the y vector without canonicalization
        y.noalias() = tr(R) * b;

        // Permute back the variables `x` to their original ordering
        x(iordering).noalias() = a;
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspace(SaddlePointSolverLegacySolveArgs args) -> void
    {
        if(isZeroMatrix(G)) solveNullspaceZeroG(args);
        else solveNullspaceDenseG(args);
    }

    /// Solve the saddle point problem using a nullspace method.
    auto solveNullspaceZeroG(SaddlePointSolverLegacySolveArgs args) -> void
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
        auto Hx    = H.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);

        // The vector y' = [ybx' ybf' ybl']
        auto yp   = vec.head(m);
        auto ypbx = yp.head(nbx);
        auto ypbf = yp.segment(nbx, nbf);
        auto ypbl = yp.tail(nl);

        // Set vectors `ax` and `af` using values from `a`
        a.noalias() = args.a(iordering);

        // Calculate b' = R*b
        b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

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
        auto x = args.x;
        auto y = args.y;

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
    auto solveNullspaceDenseG(SaddlePointSolverLegacySolveArgs args) -> void
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
        auto Hx    = H.topLeftCorner(nx, nx);
        auto Hbxbx = Hx.topLeftCorner(nbx, nbx);
        auto Hnxbx = Hx.bottomLeftCorner(nnx, nbx);

        // The sub-matrix Gbx in G' = R * G * tr(R)
        auto Gbx = G.topRows(nbx);

        // The indices of the free and fixed variables
        auto jx = iordering.head(nx);
        auto ifixed = iordering.tail(nf);

        // The right-hand side vector r = [rnx rbx rbf rbl]
        auto r = vec.head(nnx + m);
        auto rnx = r.head(nnx);
        auto rb  = r.tail(m);
        auto rbx = rb.head(nbx);
        auto rbf = rb.segment(nbx, nbf);
        auto rbl = rb.tail(nl);

        // Set vectors `ax` and `af` using values from `a`
        ax.noalias() = args.a(jx);
        af.noalias() = args.a(ifixed);

        // Calculate b' = R*b
        b.noalias() = R * args.b;

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

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
        auto x = args.x;
        auto y = args.y;

        // Calculate y = tr(R) * y'
        y.noalias() = tr(R) * rb;

        // Calculate x = [xbx xnx] using auxiliary abx and anx
        anx.noalias() = rnx;
        abx.noalias() = bbx - Sbxnx*anx - Gbx*rb;

        // Set back the values of x currently stored in a
        x(iordering).noalias() = a;
    }

    /// Calculate the relative canonical residual of equation `W*x - b`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverLegacyResidualArgs args) -> void
    {
        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Use `a` as workspace for x in the order [xbx, xnx, xbf, xnf]
        a = args.x(iordering);

        // Views to the sub-vectors of x = [xx xf]
        auto xx = a.head(nx); // the free variables in x
        auto xf = a.tail(nf); // the fixed variables in x

        // Views to the sub-vectors of xx = [xbx, xnx]
        auto xbx = xx.head(nbx); // the free basic variables in x
        auto xnx = xx.tail(nnx); // the free non-basic variables in x

        // Views to the sub-vectors of xf = [xbf, xnf]
        auto xbf = xf.head(nbf); // the fixed basic variables in x
        auto xnf = xf.tail(nnf); // the fixed non-basic variables in x

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // The indices of the free and fixed variables
        auto jx = iordering.head(nx);
        auto jf = iordering.tail(nf);

        // The relative residual vector r = [rbx rbf rbl]
        auto rbx = args.r.head(nbx);         // corresponding to free basic variables
        auto rbf = args.r.segment(nbx, nbf); // corresponding to fixed basic variables
        auto rbl = args.r.tail(nl);          // corresponding to linearly dependent equations

        //======================================================================
        // NOTE: It is extremely important to use this logic below, of
        // eliminating contribution in b from fixed variables using matrix A
        // instead of the canonical form. By doing this, we can better control
        // the feasibility error when the fixed variables correspond to
        // variables on lower bounds (i.e. 1e-40) and they can contaminate the
        // canonical residuals. If they are very small, they will either vanish
        // in the operation below using R or via the clean residual round-off
        // errors.
        //======================================================================

        // The Alias to matrix A in W = [A; J]
        const auto Wf = W(Eigen::all, jf);

        // Calculate b' = R*(b - Wf*xf)
        b.noalias() = R * (args.b - Wf*xf);

        // Ensure residual round-off errors are cleaned in b' after R*b.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(b);

        // Views to the sub-vectors of right-hand side vector b = [bbx bbf bbl]
        auto bbx = b.head(nbx);
        auto bbf = b.segment(nbx, nbf);
        auto bbl = b.tail(nl);

        // Compute rbx = xbx + Sbxnx*xnx - bbx'
        rbx.noalias() = xbx;
        rbx.noalias() += Sbxnx*xnx;
        rbx.noalias() -= bbx;

        // Normalize rbx for xbx', where xbx'[i] = xbx[i] if xbx[i] != 0 else 1
        rbx.noalias() = rbx.cwiseQuotient((xbx.array() != 0.0).select(xbx, 1.0));

        // Set the residuals to absolute values
        rbx.noalias() = rbx.cwiseAbs();

        // Set the residuals with respect to fixed basic variables to zero.
        rbf.fill(0.0);

        // Set the residuals with respect to linearly dependent equations to zero.
        rbl.fill(0.0);

        // Note: Even if there are inconsistencies above (e.g. some of these
        // residuals are not zero, like when SiO2 is fixed with 1 mol and it
        // the only species in a chemical system with element Si, but b[Si] = 2
        // mol) we consider that this is an input error and try to find a
        // solution that is feasible with respect to the free variables.
    }

    /// Calculate the relative canonical residual of equation `W*x - [b; J*x + h]`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverLegacyResidualAdvancedArgs args) -> void
    {
        // Auxiliary references
        auto [J, x, b, h, r] = args;

        // Use vec as workspace for the calculation of b' = [b; J*x + h]
        auto bprime = vec.head(m);

        bprime.head(ml) = b;
        bprime.tail(mn) = J*x + h;

        /// Calculate the relative canonical residual of equation `W*x - b'.
        residuals({ x, bprime, r });
    }

    /// Return the current state info of the saddle point solver.
    auto info() const -> SaddlePointSolverLegacyInfo
    {
        const auto jb = canonicalizer.indicesBasicVariables();
        const auto jn = canonicalizer.indicesNonBasicVariables();
        const auto S  = canonicalizer.S();
        const auto R  = canonicalizer.R();
        const auto Q  = canonicalizer.Q();
        return { jb, jn, R, S, Q };
    }
};

SaddlePointSolverLegacy::SaddlePointSolverLegacy()
: pimpl(new Impl())
{}

SaddlePointSolverLegacy::SaddlePointSolverLegacy(SaddlePointSolverLegacyInitArgs args)
: pimpl(new Impl(args))
{}

SaddlePointSolverLegacy::SaddlePointSolverLegacy(const SaddlePointSolverLegacy& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverLegacy::~SaddlePointSolverLegacy()
{}

auto SaddlePointSolverLegacy::operator=(SaddlePointSolverLegacy other) -> SaddlePointSolverLegacy&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverLegacy::setOptions(const SaddlePointOptions& options) -> void
{
    pimpl->options = options;
}

auto SaddlePointSolverLegacy::options() const -> const SaddlePointOptions&
{
    return pimpl->options;
}

auto SaddlePointSolverLegacy::canonicalize(SaddlePointSolverLegacyCanonicalizeArgs args) -> void
{
    return pimpl->canonicalize(args);
}

auto SaddlePointSolverLegacy::decompose(SaddlePointSolverLegacyDecomposeArgs args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverLegacy::solve(SaddlePointSolverLegacySolveArgs args) -> void
{
    return pimpl->solve(args);
}

auto SaddlePointSolverLegacy::solve(SaddlePointSolverLegacySolveAlternativeArgs args) -> void
{
    auto [x, y] = args;
    return pimpl->solve({x, y, x, y});
}

auto SaddlePointSolverLegacy::solve(SaddlePointSolverLegacySolveAdvancedArgs args) -> void
{
    return pimpl->solve(args);
}

auto SaddlePointSolverLegacy::residuals(SaddlePointSolverLegacyResidualArgs args) -> void
{
    return pimpl->residuals(args);
}

auto SaddlePointSolverLegacy::residuals(SaddlePointSolverLegacyResidualAdvancedArgs args) -> void
{
    return pimpl->residuals(args);
}

auto SaddlePointSolverLegacy::info() const -> SaddlePointSolverLegacyInfo
{
    return pimpl->info();
}

} // namespace Optima

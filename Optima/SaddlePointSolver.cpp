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

#include "SaddlePointSolver.hpp"

// C++ includes
#include <cassert>
#include <cmath>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/ExtendedCanonicalizer.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolverFullspace.hpp>
#include <Optima/SaddlePointSolverNullspace.hpp>
#include <Optima/SaddlePointSolverRangespace.hpp>
#include <Optima/SaddlePointTypes.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolver::Impl
{
    ExtendedCanonicalizer canonicalizer;    ///< The canonicalizer of the Jacobian matrix *W = [A; J]*.
    SaddlePointSolverRangespace rangespace; ///< The canonical saddle point solver based on a rangespace algorithm.
    SaddlePointSolverNullspace nullspace;   ///< The canonical saddle point solver based on a nullspace algorithm.
    SaddlePointSolverFullspace fullspace;   ///< The canonical saddle point solver based on a fullspace algorithm.
    SaddlePointOptions options;             ///< The options used to solve the saddle point problems.
    SaddlePointProblemDims dims;            ///< The dimensions of the saddle point problem.
    Matrix W;                               ///< The 'W = [A; J]' matrix in the saddle point matrix.
    Matrix Hw;                              ///< The workspace for Hessian matrix H.
    Vector aw;                              ///< The workspace for right-hand side vector a
    Vector bw;                              ///< The workspace for right-hand side vector b
    Vector xw;                              ///< The workspace for solution vector x
    Vector yw;                              ///< The workspace for solution vector y
    Vector weights;                         ///< The priority weights for the selection of basic variables.
    Indices iordering;                      ///< The ordering of the variables as (free-basic, free-non-basic, fixed-basic, fixed-non-basic).
    Indices Kb;                             ///< The permutation matrix used to order the basic variables as xb = (xb1, xb2, xbf) with 1 and 2 denoting pivot and non-pivot
    Indices Kn;                             ///< The permutation matrix used to order the non-basic variables as xn = (xn1, xn2, xnf) with 1 and 2 denoting pivot and non-pivot
    bool degenerate = false;                ///< The boolean flag that indicates that the decomposed saddle point matrix was degenerate with no free variables.

    /// Construct a SaddlePointSolver::Impl instance with given data.
    Impl(SaddlePointSolverInitArgs args)
    : canonicalizer(args.A),
      rangespace(args.n, args.m),
      nullspace(args.n, args.m),
      fullspace(args.n, args.m)
    {
        // Ensure consistent and proper dimensions
        assert(args.n > 0);
        assert(args.A.rows() == 0 || args.A.rows() <= args.m);
        assert(args.A.rows() == 0 || args.A.cols() == args.n);

        // Set the number of variables x and y
        const auto n = dims.n = args.n;
        const auto m = dims.m = args.m;

        // Set the number of rows in matrices *A* and *J*
        const auto ml = dims.ml = args.A.rows();
        const auto mn = dims.mn = m - ml;

        // Allocate auxiliary memory
        W.resize(m, n);
        Hw.resize(n, n);
        aw.resize(n);
        bw.resize(m);
        xw.resize(n);
        yw.resize(m);
        weights.resize(n);

        // Initialize the upper part of W = [A; J]
        W.topRows(ml) = args.A;

        // Initialize the initial ordering of the variables
        iordering = indices(n);
    }

    /// Canonicalize the *W = [A; J]* matrix of the saddle point problem.
    auto canonicalize(SaddlePointSolverCanonicalizeArgs args) -> void
    {
        // Unpack the dimension variables
        auto& [n, m, ml, mn, nb, nn, nl, nx, nf, nbx, nbf, nnx, nnf, nb1, nn1, nb2, nn2] = dims;

        // Ensure number of variables is positive.
        assert(n > 0);

        // Update the lower part of W = [A; J]
        W.bottomRows(mn) = args.J;

        // Update the number of fixed and free variables
        nf = args.jf.size();
        nx = n - nf;

        // Determine if the saddle point matrix is degenerate
        degenerate = nx == 0;

        // Skip the rest if there is no free variables
        if(degenerate)
            return;

        // The ordering of the variables as (free variables, fixed variables)
        moveIntersectionRight(iordering, args.jf);

        // The indices of the fixed (jf) variables
        const auto jf = iordering.tail(nf);

        // Update the priority weights for the update of the canonical form
        weights = args.X.array().abs();

        // Set negative priority weights for the fixed variables
        weights(jf).noalias() = -linspace(nf, 1, nf);

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

        //=========================================================================================
        // Update the order of the free variables as xx = (xb1, xb2, xbf, xn1, xn2, xnf), where:
        // -- xb1 are free pivot basic variables;
        // -- xb2 are free non-pivot basic variables;
        // -- xbf are fixed basic variables;
        // -- xn1 are free pivot non-basic variables;
        // -- xn2 are free non-pivot non-basic variables;
        // -- xnf are fixed non-basic variables.
        //-----------------------------------------------------------------------------------------
        // Note: Pivot and non-pivot as in the Gaussian elimination sense. The pivot variables
        // are those whose the corresponding diagonal entry in the Hessian matrix is not
        // dominant in the infinity norm.
        //=========================================================================================

        // Initialize the permutation matrices Kb and Kn with identity state
        Kb = indices(nb);
        Kn = indices(nn);

        // The sub-vector Kbx and Knx considering that Kb = [Kbx, Kbf] and Kn = [Knx, Knf]
        auto Kbx = Kb.head(nbx); // keep Kbf intact below to preserve position of fixed variables (already at the end!).
        auto Knx = Kn.head(nnx); // keep Knf intact below to preserve position of fixed variables (already at the end!).

        // The diagonal entries in the Hessian matrix.
        const auto Hd = args.H.diagonal();

        // The function that determines if the i-th basic variable is pivot (as in a Gaussian elimination sense).
        const auto is_pivot_basic_variable = [&](auto i) { return abs(Hd[ibasic[i]]) < 1.0; }; // TODO: The pivots should be considered in Nullspace method as well. Thus, it may make sense that the whole column of H is checked, and not just its diagonal entry.

        // The function that determines if the i-th non-basic variable is pivot (as in a Gaussian elimination sense).
        const auto is_pivot_nonbasic_variable = [&](auto i) { return abs(Hd[inonbasic[i]]) < norminf(S.col(i)); }; // TODO: The pivots should be considered in Nullspace method as well. Thus, it may make sense that the whole column of H is checked, and not just its diagonal entry.

        // Perform ordering of xbx as xbx = (xb1, xb2)
        nb1 = moveLeftIf(Kbx, is_pivot_basic_variable);

        // Perform ordering of xnx as xnx = (xn1, xn2)
        nn1 = moveLeftIf(Knx, is_pivot_nonbasic_variable);

        // Update the ordering of the basic and non-basic variables in the canonicalizer object
        canonicalizer.updateOrdering(Kb, Kn);

        // Update the number of non-pivot free basic and non-basic variables.
        nb2 = nbx - nb1;
        nn2 = nnx - nn1;

        //=========================================================================================
        // Update the order of the variables as x = (xx, xf) = (xbx, xnx, xbf, xnf), where:
        // -- xbx are free basic variables;
        // -- xnx are free non-basic variables;
        // -- xbf are fixed basic variables;
        // -- xnf are fixed non-basic variables.
        //-----------------------------------------------------------------------------------------
        // Note: By moving fixed variables away, we now have:
        // -- xbx = (xb1, xb2);
        // -- xnx = (xn1, xn2).
        //=========================================================================================

        // Get the newly ordered indices of basic and non-basic variables after the above ordering update
        const auto jb = canonicalizer.indicesBasicVariables();
        const auto jn = canonicalizer.indicesNonBasicVariables();

        // Update the ordering of the free variables as xx = [xbx xnx] = [free basic, free non-basic]
        iordering.head(nx).head(nbx) = jb.head(nbx);
        iordering.head(nx).tail(nnx) = jn.head(nnx);

        // Update the ordering of the fixed variables as xf = [xbf xnf] = [fixed basic, fixed non-basic]
        iordering.tail(nf).head(nbf) = jb.tail(nbf);
        iordering.tail(nf).tail(nnf) = jn.tail(nnf);
    }

    /// Decompose the canonical saddle point matrix.
    auto decomposeCanonical(CanonicalSaddlePointMatrix args) -> void
    {
        switch(options.method)
        {
        case SaddlePointMethod::Nullspace: nullspace.decompose(args); break;
        case SaddlePointMethod::Rangespace: rangespace.decompose(args); break;
        default: fullspace.decompose(args); break;
        }
    }

    /// Solve the canonical saddle point problem.
    auto solveCanonical(CanonicalSaddlePointProblem args) -> void
    {
        switch(options.method)
        {
        case SaddlePointMethod::Nullspace: nullspace.solve(args); break;
        case SaddlePointMethod::Rangespace: rangespace.solve(args); break;
        default: fullspace.solve(args); break;
        }
    }

    /// Decompose the coefficient matrix of the saddle point problem into canonical form.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose(SaddlePointSolverDecomposeArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto nx  = dims.nx;
        const auto nbx = dims.nbx;
        const auto nnx = dims.nnx;

        // The indices of the free variables.
        auto jx = iordering.head(nx);

        // The matrix S in the canonical form of W
        const auto S = canonicalizer.S();

        // Organize args.H in Hw in the ordering Hw = [Hxx 0; 0 0]
        auto Hxx = Hw.topLeftCorner(nx, nx);

        // Transfer the H entries to Hw taking into account the method in used
        if(options.method == SaddlePointMethod::Rangespace)
             Hxx.diagonal() = args.H.diagonal()(jx);
        else Hxx = args.H(jx, jx);

        // View to the sub-matrix Sbxnx in S = [Sbxnx Sbxnf; Sbfnx Sbfnf]
        const auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // Decompose the canonical saddle point matrix.
        decomposeCanonical({ dims, Hxx, Sbxnx });
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto nx  = dims.nx;
        const auto nf  = dims.nf;
        const auto nbx = dims.nbx;
        const auto nbf = dims.nbf;
        const auto nnx = dims.nnx;
        const auto nl  = dims.nl;

        // The matrices R and S in the canonical form of W
        const auto R = canonicalizer.R();
        const auto S = canonicalizer.S();

        // The indices of the free (jx) and fixed (jf) varaibles.
        const auto jx = iordering.head(nx);
        const auto jf = iordering.tail(nf);

        // View to the sub-matrix in W whose columns correspond to fixed variables
        const auto Wf = W(Eigen::all, jf);

        // Organize args.a in aw in the ordering aw = [ax, af]
        aw = args.a(iordering);

        // View to the sub-vectors aw = [ax, af]
        auto ax = aw.head(nx);
        auto af = aw.tail(nf);

        // Compute b' = R*(b - Wf*af)
        bw.noalias() = R * (args.b - Wf*af);

        // Ensure residual round-off errors are cleaned in b'.
        // This improves accuracy and statbility in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(bw);

        // View to the bbx sub-vector in bw = b' = R*(b - Wf*af)
        auto bbx = bw.head(nbx);

        // View to sub-matrix Hxx in Hw where H(jx, jx) has been stored during decompose
        const auto Hxx = Hw.topLeftCorner(nx, nx);

        // View to the sub-matrix Sbxnx in S = [Sbxnx Sbxnf; Sbfnx Sbfnf]
        const auto Sbxnx = S.topLeftCorner(nbx, nnx);

        // View to sub-vectors xx and xf in xw = [xx, xf]
        auto xx = xw.head(nx);
        auto xf = xw.tail(nf);

        // Set the fixed variables xf to the values in af
        xf = af;

        // View to sub-vectors ybx, ybf, ybl in yw = [ybx, ybf, ybl]
        auto ybx = yw.head(nbx);
        auto ybf = yw.segment(nbx, nbf);
        auto ybl = yw.tail(nl);

        // Set to zero the y variables w.r.t fixed basic variables
        // and linearly independent rows in matrix W = [A; J].
        ybf.fill(0.0);
        ybl.fill(0.0);

        // Solve the canonical saddle point problem.
        solveCanonical({ dims, Hxx, Sbxnx, ax, bbx, xx, ybx });

        // Compute y = tr(R) * y'
        args.y.noalias() = tr(R) * yw;

        // Transfer xw to args.x undoing the previously performed permutation
        args.x(iordering) = xw;
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAlternativeArgs args) -> void
    {
        auto [a, b] = args;
        auto [x, y] = args;
        solve({ a, b, x, y });
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAdvancedArgs args) -> void
    {
        // Unpack the data members in args
        const auto [H, J, x, g, b, h, xbar, ybar] = args;

        // Auxiliary dimension variables used below
        const auto nx = dims.nx;
        const auto nf = dims.nf;
        const auto ml = dims.ml;
        const auto mn = dims.mn;

        // The indices of the free (jx) and fixed (jf) varaibles.
        const auto jx = iordering.head(nx);
        const auto jf = iordering.tail(nf);

        // Indexed view to the sub-vectors [xx, xf] in x
        auto xx = x(jx);
        auto xf = x(jf);

        // The vectors a' = H*x - g and b' = [b, J*x - h]
        auto ap = xbar; // using xbar as workspace!
        auto bp = ybar; // using ybar as workspace!

        // Indexed view to the sub-vectors [ax', af']
        auto ax = ap(jx);
        auto af = ap(jf);

        // Indexed view to the sub-vector gx in g
        const auto gx = g(jx);

        // View to sub-matrix Hxx in Hw where H(jx, jx) has been stored during decompose
        const auto Hxx = Hw.topLeftCorner(nx, nx);

        // Compute ax' = Hxx * xx - gx
        ax = -gx;
        if(options.method == SaddlePointMethod::Rangespace)
            ax += Hxx.diagonal().cwiseProduct(xx);
        else ax += Hxx * xx;

        // Set af' = xf so that fixed variables satisfy xbar(jf) = x(jf)!
        af = xf;

        // Compute b' = [b, J*x - h]
        bp << b, J*x - h;

        // Compute the solution vectors xbar and ybar in the saddle point problem.
        solve({ ap, bp, xbar, ybar });
    }

    /// Calculate the relative canonical residual of equation `W*x - b`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto nx  = dims.nx;
        const auto nf  = dims.nf;
        const auto nbx = dims.nbx;
        const auto nbf = dims.nbf;
        const auto nnx = dims.nnx;
        const auto nnf = dims.nnf;
        const auto nl  = dims.nl;

        // Alias to the matrices of the canonicalization process
        auto S = canonicalizer.S();
        auto R = canonicalizer.R();

        // Use `aw` as workspace for x in the order [xbx, xnx, xbf, xnf]
        aw = args.x(iordering);

        // View to the sub-vectors of x = [xx xf]
        auto xx = aw.head(nx); // the free variables in x
        auto xf = aw.tail(nf); // the fixed variables in x

        // View to the sub-vectors of xx = [xbx, xnx]
        auto xbx = xx.head(nbx); // the free basic variables in x
        auto xnx = xx.tail(nnx); // the free non-basic variables in x

        // View to the sub-matrices of the canonical matrix S
        auto Sbxnx = S.topLeftCorner(nbx, nnx);
        auto Sbxnf = S.topRightCorner(nbx, nnf);
        auto Sbfnf = S.bottomRightCorner(nbf, nnf);

        // The indices of the free and fixed variables
        auto jx = iordering.head(nx);
        auto jf = iordering.tail(nf);

        // The residual vector r = [rbx rbf rbl]
        auto rbx = args.r.head(nbx);         // corresponding to free basic variables
        auto rbf = args.r.segment(nbx, nbf); // corresponding to fixed basic variables
        auto rbl = args.r.tail(nl);          // corresponding to linearly dependent equations

        // The relative residual error vector e = [ebx ebf ebl]
        auto ebx = args.e.head(nbx);         // corresponding to free basic variables
        auto ebf = args.e.segment(nbx, nbf); // corresponding to fixed basic variables
        auto ebl = args.e.tail(nl);          // corresponding to linearly dependent equations

        // The Alias to matrix A in W = [A; J]
        const auto Wf = W(Eigen::all, jf);

        //======================================================================
        // NOTE: It is extremely important to use this logic below, of
        // eliminating contribution in b from fixed variables using matrix W =
        // [A; J] instead of working on the canonical level, using matrices
        // Sbxnf and Sbfnf. instead of the canonical form. By doing this, we
        // can better control the feasibility error when the fixed variables
        // correspond to variables on lower bounds (i.e. 1e-40) and they can
        // contaminate the canonical residuals. If they are very small, they
        // will either vanish in the operation below using R or via the clean
        // residual round-off errors.
        //
        // TODO: Check if the contribution in b from fixed variables can still
        // be done at the canonical level, but the clean-round-off-errors
        // operation happens at the end of the removal process.
        //======================================================================

        // Calculate b' = R*(b - Wf*xf)
        bw.noalias() = R * (args.b - Wf*xf);

        // Ensure residual round-off errors are cleaned in b'.
        cleanResidualRoundoffErrors(bw);

        // View to the sub-vectors of right-hand side vector b = [bbx bbf bbl]
        auto bbx = bw.head(nbx);
        auto bbf = bw.segment(nbx, nbf);
        auto bbl = bw.tail(nl);

        // Compute rbx = xbx + Sbxnx*xnx - bbx'
        rbx.noalias() = xbx;
        rbx.noalias() += Sbxnx*xnx;
        rbx.noalias() -= bbx;

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

        // Compute the relative error ebx by normalizing rbx by xbx', where xbx'[i] = xbx[i] if xbx[i] != 0 else 1
        ebx.noalias() = rbx.cwiseQuotient((xbx.array() != 0.0).select(xbx, 1.0));

        // Set the errors with respect to fixed basic variables to zero.
        ebf.fill(0.0);

        // Set the errors with respect to linearly dependent equations to zero.
        ebl.fill(0.0);
    }

    /// Calculate the relative canonical residual of equation `W*x - [b; J*x + h]`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualAdvancedArgs args) -> void
    {
        // Unpack the data members in args
        auto [J, x, b, h, r, e] = args;

        // The vector b' = [b; J*x + h] (using args.r as workspace)
        auto bp = args.r;

        // Calculate b' = [b; J*x + h]
        bp << b, J*x + h;

        /// Calculate the canonical residual of equation `W*x - b'`.
        residuals({ x, bp, r, e });
    }

    /// Return the current state info of the saddle point solver.
    auto info() const -> SaddlePointSolverInfo
    {
        const auto jb = canonicalizer.indicesBasicVariables();
        const auto jn = canonicalizer.indicesNonBasicVariables();
        const auto S  = canonicalizer.S();
        const auto R  = canonicalizer.R();
        const auto Q  = canonicalizer.Q();
        return { jb, jn, R, S, Q };
    }
};

SaddlePointSolver::SaddlePointSolver(SaddlePointSolverInitArgs args)
: pimpl(new Impl(args))
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

auto SaddlePointSolver::canonicalize(SaddlePointSolverCanonicalizeArgs args) -> void
{
    return pimpl->canonicalize(args);
}

auto SaddlePointSolver::decompose(SaddlePointSolverDecomposeArgs args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolver::solve(SaddlePointSolverSolveArgs args) -> void
{
    return pimpl->solve(args);
}

auto SaddlePointSolver::solve(SaddlePointSolverSolveAlternativeArgs args) -> void
{
    auto [x, y] = args;
    return pimpl->solve({x, y, x, y});
}

auto SaddlePointSolver::solve(SaddlePointSolverSolveAdvancedArgs args) -> void
{
    return pimpl->solve(args);
}

auto SaddlePointSolver::residuals(SaddlePointSolverResidualArgs args) -> void
{
    return pimpl->residuals(args);
}

auto SaddlePointSolver::residuals(SaddlePointSolverResidualAdvancedArgs args) -> void
{
    return pimpl->residuals(args);
}

auto SaddlePointSolver::info() const -> SaddlePointSolverInfo
{
    return pimpl->info();
}

} // namespace Optima

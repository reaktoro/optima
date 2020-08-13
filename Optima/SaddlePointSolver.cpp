// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is stable software: you can redistribute it and/or modify
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
    Matrix W;                               ///< The W = [Ax Ap; Jx Jp] matrix in the saddle point matrix.
    Matrix S;                               ///< The S = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp] matrix that stores the canonical form of W = [Ax Ap; Jx Jp].
    Matrix Hw;                              ///< The workspace for H matrix.
    Vector aw;                              ///< The workspace for right-hand side vector a
    Vector bw;                              ///< The workspace for right-hand side vector b
    Vector xw;                              ///< The workspace for solution vector x
    Vector yw;                              ///< The workspace for solution vector y
    Vector weights;                         ///< The priority weights for the selection of basic variables.
    Indices iordering;                      ///< The ordering of the variables as (stable-basic, stable-non-basic, unstable-basic, unstable-non-basic).
    Indices Kb;                             ///< The permutation matrix used to order the basic variables as xb = (xbe, xbi, xbu) with `e` and `i` denoting pivot and non-pivot
    Indices Kn;                             ///< The permutation matrix used to order the non-basic variables as xn = (xne, xni, xnu) with `e` and `i` denoting pivot and non-pivot
    bool degenerate = false;                ///< The boolean flag that indicates that the decomposed saddle point matrix was degenerate with no stable variables.

    /// Construct a SaddlePointSolver::Impl instance with given data.
    Impl(SaddlePointSolverInitArgs args)
    : canonicalizer(args.Ax),
      rangespace(args.nx, args.np, args.m),
      nullspace(args.nx, args.np, args.m),
      fullspace(args.nx, args.np, args.m)
    {
        // Ensure consistent and proper dimensions
        assert(args.nx > 0);
        assert(args.Ax.rows() == 0 || args.Ax.rows() <= args.m);
        assert(args.Ap.rows() == 0 || args.Ap.rows() <= args.m);
        assert(args.Ax.rows() == 0 || args.Ax.cols() == args.nx);
        assert(args.Ap.rows() == 0 || args.Ap.cols() == args.np);

        // Initialize the number of variables x, xs, xu, p
        const auto nx = dims.nx = args.nx;
        const auto ns = dims.ns = nx;
        const auto nu = dims.nu = 0;
        const auto np = dims.np = args.np;
        const auto n  = dims.n  = nx + np;

        // Set the number of variables y, yl, yn
        const auto m  = dims.m  = args.m;
        const auto ml = dims.ml = args.Ax.rows();
        const auto mn = dims.mn = m - ml;

        // Allocate auxiliary memory
        W.resize(m, nx + np);
        S.resize(m, nx + np);
        Hw.resize(nx + np, nx + np);
        xw.resize(nx);
        yw.resize(m);
        aw.resize(nx);
        bw.resize(m);
        weights.resize(nx);

        // Initialize the upper part of W = [A; J] = [Ax Ap; Jx Jp]
        W.topLeftCorner(ml, nx) = args.Ax;
        W.topRightCorner(ml, np) = args.Ap;

        // Initialize the ordering of the variables x = (xs, xu)
        iordering = canonicalizer.Q();
    }

    /// Canonicalize the *W = [A; J]* matrix of the saddle point problem.
    auto canonicalize(SaddlePointSolverCanonicalizeArgs args) -> void
    {
        // Unpack the dimension variables
        auto& [n, nx, np, ns, nu, m, ml, mn, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nne, nbi, nni] = dims;

        // Update the number of variables in xu and xs, where x = (xs, xu)
        nu = args.ju.size();
        ns = nx - nu;

        // Ensure number of variables xs is positive.
        assert(ns > 0);

        // Determine if the saddle point matrix is degenerate
        degenerate = ns == 0; // is there one or more variable in xs?

        // Skip the rest if there is no stable variables
        if(degenerate)
            return;

        // Update the lower part of W = [A; J] = [Ax Ap; Jx Jp]
        W.bottomLeftCorner(mn, nx) = args.Jx;
        W.bottomRightCorner(mn, np) = args.Jp;

        // Change the ordering of the variables as x = (xs, xu)
        const auto pos = moveIntersectionRight(iordering, args.ju);

        // Ensure the indices of xu variables are valid.
        Assert(pos == ns, "Cannot proceed with SaddlePointSolver::canonicalize",
            "There are invalid indices of xu variables.");

        // The indices of the xs and xu variables using iordering
        const auto js = iordering.head(ns);
        const auto ju = iordering.tail(nu);

        // The priority weights of xs and xu variables to become basic variables
        auto ws = weights(js);
        auto wu = weights(ju);

        // Update the priority weights for the update of the canonical form
        ws = abs(args.wx(js));

        // Set the priority weights for the unstable variables xu to negative values.
        // This is done to prevent (as much as possible) the unstable variables
        // in *xu* from becoming basic variables if there are stable variables
        // in *xs* that can be basic instead.
        wu = -linspace(nu, 1, nu);

        // Update the canonical form of W = [A; J]
        canonicalizer.updateWithPriorityWeights(args.Jx, weights);

        // Remove residual round-off errors in Sbn after canonical form update.
        // This is an important step to ensure that residual coefficients such
        // as 1.23456e-16 are not present in matrices R and Sbn of the new
        // canonical form R*Ax*Q = [Ibx Sbn].
        canonicalizer.cleanResidualRoundoffErrors();

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = canonicalizer.indicesBasicVariables();
        const auto inonbasic = canonicalizer.indicesNonBasicVariables();

        // Update the number of basic and non-basic variables
        nb = canonicalizer.numBasicVariables();
        nn = canonicalizer.numNonBasicVariables();

        // Update the number of linearly dependent rows in W = [A; J]
        nl = m - nb;

        // Find the number of basic variables in xu (those with weights below zero)
        // Walk from last to first basic variable, since the unstable basic variables are at the end of ibasic
        for(auto i = 1; i <= nb; ++i)
            if(weights[ibasic[nb - i]] < 0.0) ++nbu;
            else break;

        // Find the number of non-basic variables in xu (those with weights below zero)
        // Walk from last to first non-basic variable, since the unstable non-basic variables are at the end of inonbasic
        for(auto i = 1; i <= nn; ++i)
            if(weights[inonbasic[nn - i]] < 0.0) ++nnu;
            else break;

        // Update the number of stable basic and stable non-basic variables
        nbs = nb - nbu;
        nns = nn - nnu;

        //=========================================================================================
        // Update the order of the stable-variables as xs = (xbe, xbi, xbu, xne, xni, xnu), where:
        // -- xbe are stable explicit basic variables (pivots);
        // -- xbi are stable implicit basic variables (non-pivots);
        // -- xbu are unstable basic variables;
        // -- xne are stable explicit non-basic variables (pivots);
        // -- xni are stable implicit non-basic variables (non-pivots);
        // -- xnu are unstable non-basic variables.
        //-----------------------------------------------------------------------------------------
        // Note: Pivot and non-pivot as in the Gaussian elimination sense. The pivot variables
        // are those whose the corresponding diagonal entry in the H matrix is not
        // dominant in the infinity norm.
        //=========================================================================================

        // Initialize the permutation matrices Kb and Kn with identity state
        Kb = indices(nb);
        Kn = indices(nn);

        // The sub-vectors Kbs and Kns in Kb = [Kbs, Kbu] and Kn = [Kns, Knu, Knp]
        // Note: Keep Kbu intact below to preserve position of xu basic
        // variables (already at the end!). The same applies for Knu and Knp,
        // since xnp variables are already at the very end (because of -inf
        // priority weights) and the xnu variables before xnp variables and
        // after xns variables.
        auto Kbs = Kb.head(nbs);
        auto Kns = Kn.head(nns);

        // The diagonal entries in the Hxx matrix.
        const auto Hd = args.Hxx.diagonal();

        // Sort the basic and non-basic stable-variables in decreasing order w.r.t. absolute values of H diagonal entries.
        std::sort(Kbs.begin(), Kbs.end(), [&](auto l, auto r) { return abs(Hd[ibasic[l]]) > abs(Hd[ibasic[r]]); }); // Note: Use > for stricter comparison, not >=!
        std::sort(Kns.begin(), Kns.end(), [&](auto l, auto r) { return abs(Hd[inonbasic[l]]) > abs(Hd[inonbasic[r]]); }); // Note: Use > for stricter comparison, not >=!

        // Update the ordering of the basic and non-basic variables in the canonicalizer object
        canonicalizer.updateOrdering(Kb, Kn);

        // Get the newly ordered indices of basic and non-basic variables after the above ordering update
        const auto jb = canonicalizer.indicesBasicVariables();
        const auto jn = canonicalizer.indicesNonBasicVariables();

        // The Wp matrix in W = [Wx Wp] = [Ax Ap; Jx Jp]
        const auto Wp = W.rightCols(np);

        // View to the sub-matrices Sbn and Sbp in S = [Sbn Sbp] = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp]
        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        // The top nb rows of R = [Rt; 0l] where 0l is present in case of rank deficiency of Ax.
        const auto Rt = canonicalizer.R().topRows(nb);

        // Update Sbn from the just canonicalized matrx Wx = [Ax; Jx]
        Sbn = canonicalizer.S();

        // Compute the Sbp matrix to complete the canonicalization of W = [Ax Ap; Jx Jp], i.e. R*W*Q' = [Ibb Sbn Sbp]
        Sbp = Rt * Wp;

        // Remove also residual round-off errors from Sbp.
        cleanResidualRoundoffErrors(Sbp);

        // Return true if the i-th basic variable is a pivot/explicit variable
        const auto is_basic_explicit = [&](auto i)
        {
            const auto idx = jb[i];                     // the global index of the basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = 1.0;                        // the max value along the corresponding column of the identity matrix
            const auto a2 = norminf(args.Hxx.col(idx)); // the max value along the corresponding column of the Hxx matrix
            const auto a3 = norminf(args.Hpx.col(idx)); // the max value along the corresponding column of the Hpx matrix
            return abs(Hii) >= std::max({a1, a2, a3});  // return true if diagonal entry is dominant
        };

        // Return true if the i-th non-basic variable is a pivot/explicit variable
        const auto is_nonbasic_explicit = [&](auto i)
        {
            const auto idx = jn[i];                     // the global index of the non-basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = norminf(Sbn.col(i));        // the max value along the corresponding column of the Sbn matrix
            const auto a2 = norminf(args.Hxx.col(idx)); // the max value along the corresponding column of the Hxx matrix
            const auto a3 = norminf(args.Hpx.col(idx)); // the max value along the corresponding column of the Hpx matrix
            return abs(Hii) >= std::max({a1, a2, a3});  // return true if diagonal entry is dominant
        };

        // Find the number of pivot basic variables (those with |Hbebe| >= I and |Hbebe| >= |cols(H, jbe)|)
        // Walk from first to last stable basic variable, since they are ordered in decresiang order of |Hii| values
        nbe = 0; while(nbe < nbs && is_basic_explicit(nbe)) ++nbe;

        // Find the number of pivot non-basic variables (those with |Hnene| >= |Sbsne| and |Hnene| > |cols(H, jne)|)
        // Walk from first to last stable non-basic variable, since they are ordered in decresiang order of |Hii| values
        nne = 0; while(nne < nns && is_nonbasic_explicit(nne)) ++nne;

        // Update the number of non-pivot/implicit stable basic and non-basic variables.
        nbi = nbs - nbe;
        nni = nns - nne;

        //=========================================================================================
        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu), where:
        // -- xbs are basic xs variables;
        // -- xns are non-basic xs variables;
        // -- xbu are basic xu variables;
        // -- xnu are non-basic xu variables.
        //-----------------------------------------------------------------------------------------
        // Note: By moving unstable-variables away, we now have:
        // -- xbs = (xbe, xbi);
        // -- xns = (xne, xni).
        //=========================================================================================

        // The indices of the basic variables in xs and xu (jbs and jbu respectively)
        const auto jbs = jb.head(nbs);
        const auto jbu = jb.tail(nbu);

        // The indices of the non-basic variables in xs and xu (jns and jnu respectively)
        const auto jns = jn.head(nns);
        const auto jnu = jn.tail(nnu);

        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu)
        iordering << jbs, jns, jbu, jnu;
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
        const auto nb  = dims.nb;
        const auto ns  = dims.ns;
        const auto np  = dims.np;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;

        // The indices of the xs variables in x = (xs, xu).
        const auto js = iordering.head(ns);

        // View to the sub-matrices Sbsns and Sbsnp in S = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp]
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsnp = S.topRightCorner(nbs, np);

        // Views to the sub-matrices Hss, Hsp, Hps, Hpp
        auto Hww = Hw.topLeftCorner(ns + np, ns + np);
        auto Hss = Hww.topLeftCorner(ns, ns);
        auto Hsp = Hww.topRightCorner(ns, np);
        auto Hps = Hww.bottomLeftCorner(np, ns);
        auto Hpp = Hww.bottomRightCorner(np, np);

        // Transfer the Hxx, Hxp, Hpx, Hpp entries to Hw taking into account the method in use
        if(options.method == SaddlePointMethod::Rangespace)
             Hss.diagonal() = args.Hxx.diagonal()(js);
        else Hss = args.Hxx(js, js);

        Hsp = rows(args.Hxp, js);
        Hps = cols(args.Hpx, js);
        Hpp = args.Hpp;

        // Decompose the canonical saddle point matrix.
        decomposeCanonical({ dims, Hss, Hsp, Hps, Hpp, Sbsns, Sbsnp });
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto np  = dims.np;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        // Unpack references in args
        auto [ax, ap, b, x, p, y] = args;

        // The canonicalizer matrix R in the canonical form of W
        const auto R = canonicalizer.R();

        // The indices of the xs, xu variables where x = (xs, xu)
        const auto js = iordering.head(ns);
        const auto ju = iordering.tail(nu);

        // View to the sub-matrix of W corresponding to xu variables
        const auto Wu = cols(W, ju);

        // Organize args.a in aw in the ordering aw = (as, au))
        aw = ax(iordering);

        // View to the sub-vectors aw = (as, au) and ap
        auto as = aw.head(ns);
        auto au = aw.tail(nu);

        // Compute b' = R*(b - Wu*au)
        bw.noalias() = R * (b - Wu * au);

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

        // View to the bbs sub-vector in bw = b' = R*(b - Wu*au)
        auto bbs = bw.head(nbs);

        // View to the sub-matrices Sbsns and Sbsnp in S = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp]
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsnp = S.topRightCorner(nbs, np);

        // Views to the sub-matrices Hss, Hsp, Hps, Hpp
        auto Hww = Hw.topLeftCorner(ns + np, ns + np);
        auto Hss = Hww.topLeftCorner(ns, ns);
        auto Hsp = Hww.topRightCorner(ns, np);
        auto Hps = Hww.bottomLeftCorner(np, ns);
        auto Hpp = Hww.bottomRightCorner(np, np);

        // View to sub-vectors in xw = (xs, xu)
        auto xs = xw.head(ns);
        auto xu = xw.tail(nu);

        // Set the xu to the values in au
        xu = au;

        // View to sub-vectors ybs, ybu, ybl in yw = [ybs, ybu, ybl]
        auto ybs = yw.head(nbs);
        auto ybu = yw.segment(nbs, nbu);
        auto ybl = yw.tail(nl);

        // Set to zero the y variables w.r.t unstable basic variables
        // and linearly independent rows in matrix W = [A; J].
        ybu.fill(0.0);
        ybl.fill(0.0);

        // Solve the canonical saddle point problem.
        solveCanonical({ dims, Hss, Hsp, Hps, Hpp, Sbsns, Sbsnp, as, ap, bbs, xs, p, ybs });

        // Compute y = tr(R) * y'
        y.noalias() = tr(R) * yw;

        // Transfer xw to x undoing the previously performed permutation
        x(iordering) = xw;
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAlternativeArgs args) -> void
    {
        auto [ax, ap, b] = args;
        auto [x, p, y] = args;
        solve({ ax, ap, b, x, p, y });
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAdvancedArgs args) -> void
    {
        // Unpack the data members in args
        const auto [x, p, g, v, b, h, xbar, pbar, ybar] = args;

        // Auxiliary dimension variables used below
        const auto nx = dims.nx;
        const auto ns = dims.ns;
        const auto nu = dims.nu;
        const auto np = dims.np;
        const auto ml = dims.ml;
        const auto mn = dims.mn;

        // The indices of the xs and xu variables
        const auto js = iordering.head(ns);
        const auto ju = iordering.tail(nu);

        // Indexed view to the sub-vectors in x = (xs, xu)
        auto xs = x(js);
        auto xu = x(ju);

        // The vectors ax' = H*x - g and bw = [b, Jx*x + Jp*p - h]
        auto ax = xbar; // using xbar as workspace!
        auto ap = pbar; // using pbar as workspace!
        auto bw = ybar; // using ybar as workspace!

        // Indexed view to the sub-vectors ax = (as, au)
        auto as = ax(js);
        auto au = ax(ju);

        // Indexed view to the sub-vector gs in g = (gs, gu) where g = fx
        const auto gs = g(js);

        // Views to the sub-matrices Hss, Hsp, Hps, Hpp
        const auto Hww = Hw.topLeftCorner(ns + np, ns + np);
        const auto Hss = Hww.topLeftCorner(ns, ns);
        const auto Hsp = Hww.topRightCorner(ns, np);
        const auto Hps = Hww.bottomLeftCorner(np, ns);
        const auto Hpp = Hww.bottomRightCorner(np, np);

        // Compute as = Hss * xs + Hsp * p - gs
        as = Hsp * p - gs;
        if(options.method == SaddlePointMethod::Rangespace)
            as += Hss.diagonal().cwiseProduct(xs);
        else as += Hss * xs;

        // Compute ap = Hps * xs + Hpp * p - v
        ap = Hps * xs + Hpp * p - v;

        // Set au = xu so that fixed/unstable variables satisfy xbar(ju) = x(ju)!
        au = xu;

        // Views to sub-matrices Jx and Jp in W = [Ax Ap; Jx Jp]
        const auto Jx = W.bottomLeftCorner(mn, nx);
        const auto Jp = W.bottomRightCorner(mn, np);

        // Compute bw = (b, Jx * x + Jp * p - h)
        bw << b, Jx*x + Jp*p - h;

        // Compute the solution vectors xbar and ybar in the saddle point problem.
        solve({ ax, ap, bw, xbar, pbar, ybar });
    }

    /// Calculate the relative canonical residual of equation `W*x - b`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto nx  = dims.nx;
        const auto np  = dims.np;
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nnu = dims.nnu;
        const auto nl  = dims.nl;

        // The canonicalizer matrix R in the canonical form of W
        const auto R = canonicalizer.R();

        // Use `aw` as workspace for x in the order [xbs, xns, xbu, xnu]
        aw = args.x(iordering);

        // View to the sub-vectors of x = (xs, xu)
        auto xs = aw.head(ns);
        auto xp = aw.tail(nu);
        auto xu = aw.tail(nu);

        // View to the sub-vectors of xs = [xbs, xns]
        auto xbs = xs.head(nbs);
        auto xns = xs.tail(nns);

        // View to the sub-matrices Sbsns and Sbsnp in S = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp]
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsnp = S.topRightCorner(nbs, np);

        // The indices of the xs, xu variables where x = (xs, xu)
        const auto js = iordering.head(ns);
        const auto ju = iordering.tail(nu);

        // The residual vector r = [rbs rbu rbl]
        auto rbs = args.r.head(nbs);         // corresponding to stable basic variables
        auto rbu = args.r.segment(nbs, nbu); // corresponding to unstable basic variables
        auto rbl = args.r.tail(nl);          // corresponding to linearly dependent equations

        // The relative residual error vector e = [ebs ebu ebl]
        auto ebs = args.e.head(nbs);         // corresponding to free basic variables
        auto ebu = args.e.segment(nbs, nbu); // corresponding to fixed basic variables
        auto ebl = args.e.tail(nl);          // corresponding to linearly dependent equations

        // View to the sub-matrix of W corresponding to xu variables
        const auto Wu = cols(W, ju);

        //======================================================================
        // NOTE: It is extremely important to use this logic below, of
        // eliminating contribution in b from unstable-variables using matrix W =
        // [A; J] instead of working on the canonical level, using matrices
        // Sbsnu and Sbfnf. instead of the canonical form. By doing this, we
        // can better control the feasibility error when the unstable-variables
        // correspond to variables on lower bounds (i.e. 1e-40) and they can
        // contaminate the canonical residuals. If they are very small, they
        // will either vanish in the operation below using R or via the clean
        // residual round-off errors.
        //
        // TODO: Check if the contribution in b from unstable-variables can still
        // be done at the canonical level, but the clean-round-off-errors
        // operation happens at the end of the removal process.
        //======================================================================

        // Compute b' = R*(b - Wu*xu)
        bw.noalias() = R * (args.b - Wu * xu);

        // Ensure residual round-off errors are cleaned in b'.
        cleanResidualRoundoffErrors(bw);

        // View to the sub-vectors of right-hand side vector b = [bbs bbf bbl]
        auto bbs = bw.head(nbs);
        auto bbf = bw.segment(nbs, nbu);
        auto bbl = bw.tail(nl);

        // Compute rbs = xbs + Sbsns*xns + Sbsnp*xnp - bbs'
        rbs.noalias() = xbs;
        rbs.noalias() += Sbsns*xns;
        rbs.noalias() += Sbsnp*xp;
        rbs.noalias() -= bbs;

        // Set the residuals to absolute values
        rbs.noalias() = rbs.cwiseAbs();

        // Set the residuals with respect to unstable basic variables to zero.
        rbu.fill(0.0);

        // Set the residuals with respect to linearly dependent equations to zero.
        rbl.fill(0.0);

        // Note: Even if there are inconsistencies above (e.g. some of these
        // residuals are not zero, like when SiO2 is unstable with 1 mol and it
        // the only species in a chemical system with element Si, but b[Si] = 2
        // mol) we consider that this is an input error and try to find a
        // solution that is feasible with respect to the stable-variables.

        // Compute the relative error ebs by normalizing rbs by xbs', where xbs'[i] = xbs[i] if xbs[i] != 0 else 1
        ebs.noalias() = rbs.cwiseQuotient((xbs.array() != 0.0).select(xbs, 1.0));

        // Set the errors with respect to unstable basic variables to zero.
        ebu.fill(0.0);

        // Set the errors with respect to linearly dependent equations to zero.
        ebl.fill(0.0);
    }

    /// Calculate the relative canonical residual of equation `W*x - [b; J*x + h]`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualAdvancedArgs args) -> void
    {
        // Auxiliary dimension variables used below
        const auto nx = dims.nx;
        const auto np = dims.np;
        const auto mn = dims.mn;

        // Unpack the data members in args
        auto [x, p, b, h, r, e] = args;

        // The vector bw = [b; Jx*x + Jp*p + h] (using args.r as workspace)
        auto bw = args.r;

        // Views to sub-matrices Jx and Jp in W = [Ax Ap; Jx Jp]
        const auto Jx = W.bottomLeftCorner(mn, nx);
        const auto Jp = W.bottomRightCorner(mn, np);

        // Compute bw = (b, Jx * x + Jp * p - h)
        bw << b, Jx*x + Jp*p - h;

        /// Calculate the canonical residual of equation `W*x - b'`.
        residuals({ x, p, bw, r, e });
    }

    /// Calculate the multiplication of the saddle point matrix with a vector *(x, y)*.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto multiply(SaddlePointSolverMultiplyArgs args) -> void
    {
        // // Unpack the data members in args
        // const auto [x, y, a, b] = args;

        // // Auxiliary dimension variables used below
        // const auto nx  = dims.nx;
        // const auto np  = dims.np;
        // const auto nu  = dims.nu;
        // const auto nbs = dims.nbs;
        // const auto nbu = dims.nbu;
        // const auto nns = dims.nns;
        // const auto nnu = dims.nnu;
        // const auto nl  = dims.nl;

        // // Alias to the matrices of the canonicalization process
        // auto S = canonicalizer.S();
        // auto R = canonicalizer.R();

        // // The indices of the xs, xu variables in x = (xs, xu)
        // auto js = iordering.head(ns);
        // auto ju = iordering.tail(nu);
        // // auto jp = iordering.tail(np); TODO: Check this <<<<<<<<<<------------------------------

        // // Use `xw` as workspace for x in the order (xs, xu)
        // xw = args.x(iordering);

        // // View to sub-vectors in xw = (xs, xu)
        // const auto xs = xw.head(ns);
        // const auto xp = xw.tail(nu);
        // const auto xu = xw.tail(nu);

        // // View to the sub-matrices Hxx, Hpx, Hpp in Hw = [Hxx 0 0; Hpx Hpp 0; 0 0 0]
        // const auto Hxx = Hw.topLeftCorner(nx, nx);
        // const auto Hpx = Hw.middleRows(nx, np).leftCols(nx);
        // const auto Hpp = Hw.middleRows(nx, np).middleCols(nx, np);

        // // View to the sub-matrices Sbsns and Sbsnp in S = [Sbsns Sbsnu Sbsnp; 0 Sbunu Sbunp]
        // const auto Sbsns = S.topLeftCorner(nbs, nns);
        // const auto Sbsnp = S.topRightCorner(nbs, np);

        // // View to the sub-vectors of xs = [xbs, xns]
        // auto xbs = xs.head(nbs);
        // auto xns = xs.tail(nns);

        // // View to the bbs sub-vector in bw = b' = R*(b - Wu*au)
        // auto bbs = bw.head(nbs);

        // // Set the unstable-variables xu to the values in au
        // xu = au;

        // // View to sub-vectors ybs, ybu, ybl in yw = [ybs, ybu, ybl]
        // auto ybs = yw.head(nbs);
        // auto ybu = yw.segment(nbs, nbu);
        // auto ybl = yw.tail(nl);

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
    auto [x, p, y] = args;
    return pimpl->solve({x, p, y, x, p, y});
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

auto SaddlePointSolver::multiply(SaddlePointSolverMultiplyArgs args) -> void
{
    return pimpl->multiply(args);
}

auto SaddlePointSolver::info() const -> SaddlePointSolverInfo
{
    return pimpl->info();
}

} // namespace Optima

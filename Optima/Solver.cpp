// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

#include "Solver.hpp"

// Optima includes
#include <Optima/Constants.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/MasterSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Problem.hpp>
#include <Optima/Result.hpp>
#include <Optima/State.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {
namespace {

auto toMasterDims(const Dims& dims) -> MasterDims
{
    const auto [nx, np, nbe, nbg, nhe, nhg] = dims;

    const auto nr   = nbg;
    const auto ns   = nhg;
    const auto nxrs = nx + nr + ns;
    const auto ny   = nbe + nbg;
    const auto nz   = nhe + nhg;

    return MasterDims(nxrs, np, ny, nz);
}

} // namespace

struct Solver::Impl
{
    MasterSolver msolver;   ///< The master optimization solver.
    MasterProblem mproblem; ///< The master optimization problem.
    Index nx   = 0;         ///< The number of variables x in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index nr   = 0;         ///< The number of variables r in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index ns   = 0;         ///< The number of variables s in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index nxrs = 0;         ///< The number of variables in xrs = (x, xbg, xhg).
    Index np   = 0;         ///< The number of parameter variables p.
    Index ny   = 0;         ///< The number of Lagrange multipliers y (i.e., the dimension of vector b = (be, bg)).
    Index nz   = 0;         ///< The number of Lagrange multipliers z (i.e., the dimension of vector h = (he, hg)).
    Vector xrslower;        ///< The lower bounds of vector xrs = (x, xbg, xhg) in the master optimization problem.
    Vector xrsupper;        ///< The upper bounds of vector xrs = (x, xbg, xhg) in the master optimization problem.

    /// Construct a Solver default instance.
    Impl()
    {
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void
    {
        msolver.setOptions(options);
    }

    /// Solve the optimization problem.
    auto solve(const Problem& problem, State& state) -> Result
    {
        // Auxiliary references
        const auto& dims = problem.dims;

        // Initialize dimension variables
        nx   = dims.x;
        nr   = dims.bg;
        ns   = dims.hg;
        nxrs = nx + nr + ns;
        np   = dims.p;
        ny   = dims.be + dims.bg;
        nz   = dims.he + dims.hg;

        error(!problem.f.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the objective function. "
            "Ensure Problem::f is properly initialized.");

        error(dims.he > 0 && !problem.he.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the constraint function he(x, p). "
            "Ensure Problem::he is properly initialized.");

        error(dims.hg > 0 && !problem.hg.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the constraint function hg(x, p). "
            "Ensure Problem::hg is properly initialized.");

        error(dims.p > 0 && !problem.v.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the complementary constraint function v(x, p). "
            "Ensure Problem::v is properly initialized.");

        // Initialize the dimensions of the master optimization problem
        mproblem.dims = toMasterDims(dims);

        // Create the objective function for the master optimization problem
        mproblem.f = [&](ObjectiveResultRef res, VectorView xrs, VectorView p, ObjectiveOptions opts)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-vectors in fxrs = (fx, fr, fs)
            auto fx = res.fx.head(nx);
            auto fr = res.fx.segment(nx, nr);
            auto fs = res.fx.tail(ns);

            // Views to sub-matrices in fxrsxrs = [ [fxx fxr fxs], [frx frr frs], [fsx fsr fss] ]
            auto fxx = res.fxx.topRows(nx).leftCols(nx);
            auto fxr = res.fxx.topRows(nx).middleCols(nx, nr);
            auto fxs = res.fxx.topRows(nx).rightCols(ns);

            auto frx = res.fxx.middleRows(nx, nr).leftCols(nx);
            auto frr = res.fxx.middleRows(nx, nr).middleCols(nx, nr);
            auto frs = res.fxx.middleRows(nx, nr).rightCols(ns);

            auto fsx = res.fxx.bottomRows(ns).leftCols(nx);
            auto fsr = res.fxx.bottomRows(ns).middleCols(nx, nr);
            auto fss = res.fxx.bottomRows(ns).rightCols(ns);

            // Views to sub-matrices in fxrsp = [ [fxp], [frp], [fsp] ]
            auto fxp = res.fxp.topRows(nx);
            auto frp = res.fxp.middleRows(nx, nr);
            auto fsp = res.fxp.bottomRows(ns);

            // Set blocks to zero, except fx, fxx, fxp (computed via the objective function next)
            fr.fill(0.0);
            fs.fill(0.0);
            fxr.fill(0.0);
            fxs.fill(0.0);
            frx.fill(0.0);
            frr.fill(0.0);
            frs.fill(0.0);
            fsx.fill(0.0);
            fsr.fill(0.0);
            fss.fill(0.0);
            frp.fill(0.0);
            fsp.fill(0.0);

            // Use the objective function to compute f, fx, fxx, fxp
            ObjectiveResultRef fres(res.f, fx, fxx, fxp, res.diagfxx, res.fxx4basicvars, res.succeeded);

            problem.f(fres, x, p, opts);
        };

        // Create the non-linear equality constraint for the master optimization problem
        mproblem.h = [&](ConstraintResultRef res, VectorView xrs, VectorView p, ConstraintOptions opts)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-vectors in h = (he, hg)
            auto he = res.val.topRows(dims.he);
            auto hg = res.val.bottomRows(dims.hg);

            // Views to sub-matrices in dh/d(xrs) = [ [he_x he_r he_s], [hg_x hg_r hg_s] ]
            auto he_x = res.ddx.topRows(dims.he).leftCols(nx);
            auto he_r = res.ddx.topRows(dims.he).middleCols(nx, nr);
            auto he_s = res.ddx.topRows(dims.he).rightCols(ns);

            auto hg_x = res.ddx.bottomRows(dims.hg).leftCols(nx);
            auto hg_r = res.ddx.bottomRows(dims.hg).middleCols(nx, nr);
            auto hg_s = res.ddx.bottomRows(dims.hg).rightCols(ns);

            // Views to sub-matrices in dh/dp = [he_p; hg_p]
            auto he_p = res.ddp.topRows(dims.he);
            auto hg_p = res.ddp.bottomRows(dims.hg);

            // Set all blocks to zero, except hg_s which is I and he_x and hg_x computed next via he and hg functions
            he_r.fill(0.0);
            he_s.fill(0.0);
            hg_r.fill(0.0);
            hg_s.fill(0.0);
            hg_s.diagonal().fill(1.0);

            ConstraintResultRef re(he, he_x, he_p, res.ddx4basicvars, res.succeeded);

            problem.he(re, x, p, opts);

            ConstraintResultRef rg(hg, hg_x, hg_p, res.ddx4basicvars, res.succeeded);

            problem.hg(rg, x, p, opts);

            hg.noalias() += s;
        };

        // Create the external non-linear constraint for the master optimization problem
        mproblem.v = [&](ConstraintResultRef res, VectorView xrs, VectorView p, ConstraintOptions opts)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-matrices in dv/d(xrs) = [ vx vr vs ]
            auto vx = res.ddx.leftCols(nx);
            auto vr = res.ddx.middleCols(nx, nr);
            auto vs = res.ddx.rightCols(ns);

            // Auxiliary references to v and vp = dv/dp
            auto v  = res.val;
            auto vp = res.ddp;

            // Set vr = dv/dr = 0 and vs = dv/ds = 0
            vr.fill(0.0);
            vs.fill(0.0);

            // Compute v, vx, vp using the given external constraint function v(x, p)
            ConstraintResult vres(v, vx, vp, res.ddx4basicvars, res.succeeded);

            problem.v(vres, x, p, opts);
        };

        // Initialize vector with lower bounds for bar(x) = (x, xbg, xhg)
        xrslower.resize(nxrs);
        xrslower.head(nx) = problem.xlower;
        xrslower.tail(nr + ns).fill(-infinity());

        // Initialize vector with upper bounds for bar(x) = (x, xbg, xhg)
        xrsupper.resize(nxrs);
        xrsupper.head(nx) = problem.xupper;
        xrslower.tail(nr + ns).fill(0.0);

        // Initialize vector b = (be, bg)
        mproblem.b.resize(ny);
        mproblem.b << problem.be, problem.bg;

        mproblem.xlower = xrslower;
        mproblem.xupper = xrsupper;
        mproblem.plower = problem.plower;
        mproblem.pupper = problem.pupper;

        // Create matrix Ax = [ [Aex, 0, 0], [Agx, I, 0] ]
        mproblem.Ax = zeros(ny, nxrs);
        mproblem.Ax.leftCols(nx) << problem.Aex, problem.Agx;
        mproblem.Ax.middleCols(nx, nr).bottomRows(nr).diagonal().fill(1.0);

        // Create matrix Ap = [ [Aep], [Agp] ]
        mproblem.Ap = zeros(ny, np);
        if(np > 0) mproblem.Ap << problem.Aep, problem.Agp;

        // Create references to state members
        auto xbar       = state.xbar;
        auto p          = state.p;
        auto w          = state.w;
        auto sbar       = state.sbar;
        auto& stability = state.stability;

        auto result = msolver.solve(mproblem, { xbar, p, w });

        return result;
    }
};

Solver::Solver()
: pimpl(new Impl())
{}

Solver::Solver(const Solver& other)
: pimpl(new Impl(*other.pimpl))
{}

Solver::~Solver()
{}

auto Solver::operator=(Solver other) -> Solver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Solver::setOptions(const Options& options) -> void
{
	pimpl->setOptions(options);
}

auto Solver::solve(const Problem& problem, State& state) -> Result
{
    return pimpl->solve(problem, state);
}

} // namespace Optima

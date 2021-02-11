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

struct Solver::Impl
{
    MasterSolver msolver;   ///< The master optimization solver.
    MasterProblem mproblem; ///< The master optimization problem.
    MasterState mstate;     ///< The master optimization state.
    Index nx    = 0;        ///< The number of variables x in xbar = (x, xbg, xhg).
    Index nxbg  = 0;        ///< The number of variables xbg in xbar = (x, xbg, xhg).
    Index nxhg  = 0;        ///< The number of variables xhg in xbar = (x, xbg, xhg).
    Index nxbar = 0;        ///< The number of variables in xbar = (x, xbg, xhg).
    Index np    = 0;        ///< The number of parameter variables p.
    Index ny    = 0;        ///< The number of Lagrange multipliers y (i.e., the dimension of vector b = (be, bg)).
    Index nz    = 0;        ///< The number of Lagrange multipliers z (i.e., the dimension of vector h = (he, hg)).
    Index nwbar = 0;        ///< The number of Lagrange multipliers in wbar = (ye, yg, ze, zg).
    Vector xbar;            ///< The vector xbar = (x, u, v) in the master optimization problem.
    Vector wbar;            ///< The vector wbar = (ye, yg, ze, zg) in the master optimization problem.
    Vector xbarlower;       ///< The lower bounds of vector xbar = (x, xbg, xhg) in the master optimization problem.
    Vector xbarupper;       ///< The upper bounds of vector xbar = (x, xbg, xhg) in the master optimization problem.

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
        nx    = dims.x;
        nxbg  = dims.bg;
        nxhg  = dims.hg;
        nxbar = nx + nxbg + nxhg;
        np    = dims.p;
        ny    = dims.be + dims.bg;
        nz    = dims.he + dims.hg;
        nwbar = ny + nz;

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
        mproblem.dims = MasterDims(nxbar, np, ny, nz);

        // Create the objective function for the master optimization problem
        mproblem.f = [&](ObjectiveResultRef resbar, VectorView xbar, VectorView p, VectorView c, ObjectiveOptions opts)
        {
            resbar.fx.fill(0.0);
            resbar.fxx.fill(0.0);
            resbar.fxp.fill(0.0);
            resbar.fxc.fill(0.0);

            auto x   = xbar.head(nx);
            auto fx  = resbar.fx.head(nx);
            auto fxx = resbar.fxx.topLeftCorner(nx, nx);
            auto fxp = resbar.fxp.topRows(nx);
            auto fxc = resbar.fxc.topRows(nx);

            ObjectiveResultRef fres(resbar.f, fx, fxx, fxp, fxc, resbar.diagfxx, resbar.fxx4basicvars, resbar.succeeded);

            problem.f(fres, x, p, c, opts);
        };

        // Create the non-linear equality constraint for the master optimization problem
        mproblem.h = [&](ConstraintResultRef resbar, VectorView xbar, VectorView p, VectorView c, ConstraintOptions opts)
        {
            // Views to sub-vectors in xbar = (x, xbg, xhg)
            const auto x   = xbar.head(nx);
            const auto xbg = xbar.segment(nx, nxbg);
            const auto xhg = xbar.tail(nxhg);

            // Views to sub-vectors in h = (he, hg)
            auto he = resbar.val.head(dims.he);
            auto hg = resbar.val.tail(dims.hg);

            // Views to sub-matrices in dh/d(xbar) = [ [dhe/dx dhe/dxbg dhe/dxhg], [dhg/dx dhg/dxbg dhg/dxhg] ]
            auto he_x   = resbar.ddx.topRows(dims.he).leftCols(nx);
            auto he_xbg = resbar.ddx.topRows(dims.he).middleCols(nx, nxbg);
            auto he_xhg = resbar.ddx.topRows(dims.he).rightCols(nxhg);

            auto hg_x   = resbar.ddx.bottomRows(dims.hg).leftCols(nx);
            auto hg_xbg = resbar.ddx.bottomRows(dims.hg).middleCols(nx, nxbg);
            auto hg_xhg = resbar.ddx.bottomRows(dims.hg).rightCols(nxhg);

            // Views to sub-matrices in dh/dp = [he_p; hg_p]
            auto he_p = resbar.ddp.topRows(dims.he);
            auto hg_p = resbar.ddp.bottomRows(dims.hg);

            // Views to sub-matrices in dh/dc = [he_c; hg_c]
            auto he_c = resbar.ddc.topRows(dims.he);
            auto hg_c = resbar.ddc.bottomRows(dims.hg);

            // Set all blocks related to xbg and xhg to zero, except hg_xhg which is identity
            he_xbg.fill(0.0);
            he_xhg.fill(0.0);
            hg_xbg.fill(0.0);
            hg_xhg.fill(0.0);
            hg_xhg.diagonal().fill(1.0);

            ConstraintResultRef heres(he, he_x, he_p, he_c, resbar.ddx4basicvars, resbar.succeeded);

            problem.he(heres, x, p, c, opts);

            ConstraintResultRef hgres(hg, hg_x, hg_p, hg_c, resbar.ddx4basicvars, resbar.succeeded);

            problem.hg(hgres, x, p, c, opts);

            hg.noalias() += xhg;
        };

        // Create the external non-linear constraint for the master optimization problem
        mproblem.v = [&](ConstraintResultRef res, VectorView xbar, VectorView p, VectorView c, ConstraintOptions opts)
        {
            // Views to sub-vectors in xbar = (x, xbg, xhg)
            const auto x   = xbar.head(nx);
            const auto xbg = xbar.segment(nx, nxbg);
            const auto xhg = xbar.tail(nxhg);

            // Views to sub-matrices in dv/d(xbar) = [ dv/dx dv/dxbg dv/dxhg ]
            auto v_x   = res.ddx.leftCols(nx);
            auto v_xbg = res.ddx.middleCols(nx, nxbg);
            auto v_xhg = res.ddx.rightCols(nxhg);

            // Auxiliary references to v, vp = dv/dp, vc = dv/dc
            auto v   = res.val;
            auto v_p = res.ddp;
            auto v_c = res.ddc;

            // Set dv/dxbg = 0 and dv/dxhg = 0
            v_xbg.fill(0.0);
            v_xhg.fill(0.0);

            ConstraintResult vres(v, v_x, v_p, v_c, res.ddx4basicvars, res.succeeded);

            problem.v(vres, x, p, c, opts);
        };

        // Initialize xbar = (x, xbg, xhg)
        xbar.resize(nxbar);
        xbar << state.x, state.xbg, state.xhg;

        // Initialize wbar = (ye, yg, ze, zg)
        wbar.resize(nwbar);
        wbar << state.ye, state.yg, state.ze, state.zg;

        // Initialize the lower bounds of xbar = (x, xbg, xhg)
        xbarlower.resize(nxbar);
        xbarlower.head(nx) = problem.xlower;
        xbarlower.tail(nxbg + nxhg).fill(-infinity());

        // Initialize the upper bounds of xbar = (x, xbg, xhg)
        xbarupper.resize(nxbar);
        xbarupper.head(nx) = problem.xupper;
        xbarlower.tail(nxbg + nxhg).fill(0.0);

        // Initialize vector b = (be, bg)
        mproblem.b.resize(ny);
        mproblem.b << problem.be, problem.bg;

        mproblem.xlower = xbarlower;
        mproblem.xupper = xbarupper;
        mproblem.plower = problem.plower;
        mproblem.pupper = problem.pupper;

        // Create matrix Ax = [ [Aex, 0, 0], [Agx, I, 0] ]
        mproblem.Ax.resize(ny, nxbar);
        if(mproblem.Ax.size()) {
            mproblem.Ax.leftCols(nx) << problem.Aex, problem.Agx;
            mproblem.Ax.middleCols(nx, nxbg).bottomRows(nxbg) = identity(nxbg, nxbg);
            mproblem.Ax.rightCols(nxhg).fill(0.0);
        }

        // Create matrix Ap = [ [Aep], [Agp] ]
        mproblem.Ap.resize(ny, np);
        if(mproblem.Ap.size())
            mproblem.Ap << problem.Aep, problem.Agp;

        mstate.u.resize(mproblem.dims);
        mstate.u.x = xbar;
        mstate.u.p = state.p;
        mstate.u.w = wbar;

        // Perform the master optimization calculation
        // auto result = msolver.solve(mproblem, { xbar, state.p, wbar });
        auto result = msolver.solve(mproblem, mstate);

        xbar    = mstate.u.x;
        state.p = mstate.u.p;
        wbar    = mstate.u.w;

        // Transfer computed xbar = (x, xbg, xhg) to state
        state.x   = xbar.head(nx);
        state.xbg = xbar.segment(nx, nxbg);
        state.xhg = xbar.tail(nxhg);

        // Transfer computed wbar = (ye, yg, ze, zg) to state
        state.ye = wbar.head(ny).head(dims.be);
        state.yg = wbar.head(ny).tail(dims.bg);
        state.ze = wbar.tail(nz).head(dims.he);
        state.zg = wbar.tail(nz).tail(dims.hg);

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

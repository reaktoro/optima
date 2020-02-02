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

#include "Solver.hpp"

// Optima includes
#include <Optima/BasicSolver.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Options.hpp>
#include <Optima/Problem.hpp>
#include <Optima/Result.hpp>
#include <Optima/State.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {
namespace detail {

auto initBasicSolver(const Problem& problem) -> BasicSolver
{
    const auto [nx, mbe, mbg, mhe, mhg] = problem.dims;

    const auto nu = mbg;
    const auto nv = mhg;
    const auto n  = nx + nu + nv;
    const auto mb = mbe + mbg;
    const auto mh = mhe + mhg;
    const auto m  = mb + mh;

    // Create the matrix A = [ [Ae, 0, 0], [Ag, -I, 0] ]
    Matrix A = zeros(mb, n);
    A.leftCols(nx) << problem.Ae, problem.Ag;
    A.middleCols(nx, nu).bottomRows(nu).diagonal().fill(-1.0);

    return BasicSolver({n, m, A});
}

} // namespace detail

struct Solver::Impl
{
    /// The basic optimization solver.
    BasicSolver solver;

    /// The dimension information of variables and constraints in the optimization problem.
    Dims dims;

    /// The number of variables in bar(x) = (x, u, v).
    Index n = 0;

    /// The number of variables x in bar(x) = (x, u, v).
    Index nx = 0;

    /// The number of variables u in bar(x) = (x, u, v).
    Index nu = 0;

    /// The number of variables v in bar(x) = (x, u, v).
    Index nv = 0;

    /// The dimension of vector b = [be, bg].
    Index mb = 0;

    /// The dimension of vector h = [he, hg].
    Index mh = 0;

    /// The dimension of number m = mb + mh.
    Index m = 0;

    /// The vector bar(x) = [x, u, v] in the basic optimization problem.
    Vector xbar;

    /// The vector bar(z) = [zx, zu, zv] in the basic optimization problem.
    Vector zbar;

    /// The right-hand side vector b = [be, bg] in the basic optimization problem.
    Vector b;

    /// The lower bounds of vector bar(x) = [x, u, v] in the basic optimization problem.
    Vector xbar_lower;

    /// The upper bounds of vector bar(x) = [x, u, v] in the basic optimization problem.
    Vector xbar_upper;

    /// The ordering of the variables bar(x) = [x, u, v] as (*stable*, *lower unstable*, *upper unstable*).
    Indices iordering;

    /// The number of lower unstable variables in bar(x) = [x, u, v].
    IndexNumber nul;

    /// The number of upper unstable variables in bar(x) = [x, u, v].
    IndexNumber nuu;

    /// Construct a Solver instance with given optimization problem.
    Impl(const Problem& problem)
    : solver(detail::initBasicSolver(problem)), dims(problem.dims)
    {
        // Initialize dimension variables
        nx = dims.x;
        nu = dims.bg;
        nv = dims.hg;
        n  = nx + nu + nv;
        mb = dims.be + dims.bg;
        mh = dims.he + dims.hg;
        m  = mb + mh;

        // Initialize vectors
        xbar.resize(n);
        zbar.resize(n);
        xbar_lower.resize(n);
        xbar_upper.resize(n);
        b.resize(m);

        // Initialize the ordering of the variables.
        iordering = indices(n);
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void
    {
        solver.setOptions(options);
    }

    /// Solve the optimization problem.
    auto solve(StateRef state, const Problem& problem) -> Result
    {
        // Create the objective function for the basic problem
        auto f = [&](VectorConstRef xbar, ObjectiveResult& res)
        {
            const auto x = xbar.head(nx);

            auto gx = res.g.head(nx);
            auto Hx = res.H.topLeftCorner(nx, nx);

            res.H.rightCols(nu + nv).fill(0.0);
            res.H.bottomLeftCorner(nu + nv, nx).fill(0.0);

            ObjectiveResult r{res.f, gx, Hx};
            r.requires = res.requires;

            problem.f(x, r);

            res.failed = r.failed;
        };

        // Create the non-linear equality constraint for the basic problem
        auto h = [&](VectorConstRef xbar, ConstraintResult& res)
        {
            const auto x = xbar.head(nx);
            const auto v = xbar.head(nv);

            auto he = res.h.head(dims.he);
            auto hg = res.h.tail(dims.hg);
            auto Je = res.J.topLeftCorner(dims.he, nx);
            auto Jg = res.J.bottomLeftCorner(dims.hg, nx);

            res.J.rightCols(nu + nv).fill(0.0);
            res.J.bottomRightCorner(nv, nv).diagonal().fill(-1.0);

            ConstraintResult re{he, Je};
            ConstraintResult rg{hg, Jg};

            problem.he(x, re);
            problem.hg(x, rg);

            hg.noalias() -= v;

            res.failed = re.failed || rg.failed;
        };

        // Initialize vector bar(x) = [x, u, v]
        xbar.head(nx) = state.x;
        xbar.tail(nu + nv).fill(0.0);

        // Initialize vector with lower bounds for bar(x) = [x, u, v]
        xbar_lower.head(nx) = problem.xlower;
        xbar_lower.tail(nu + nv).fill(0.0);

        // Initialize vector with upper bounds for bar(x) = [x, u, v]
        xbar_upper.head(nx) = problem.xupper;
        xbar_lower.tail(nu + nv).fill(infinity());

        // Initialize vector b = [be, bg]
        b << problem.be, problem.bg;

        // Create alias for y vector
        auto y = state.y;

        // Solve the constructed basic optimization problem
        auto result = solver.solve({ f, h, b, xbar_lower, xbar_upper, xbar, y, zbar, iordering, nul, nuu });

        return result;
    }
};

Solver::Solver(const Problem& problem)
: pimpl(new Impl(problem))
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

auto Solver::solve(StateRef state, const Problem& problem) -> Result
{
    return pimpl->solve(state, problem);
}

} // namespace Optima

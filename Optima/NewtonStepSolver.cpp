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

#include "NewtonStepSolver.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/JacobianMatrix.hpp>
#include <Optima/NewtonStepSolverFullspace.hpp>
#include <Optima/NewtonStepSolverNullspace.hpp>
#include <Optima/NewtonStepSolverRangespace.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/SolutionVector.hpp>

namespace Optima {

struct NewtonStepSolver::Impl
{
    const Index nx = 0; ///< The number of variables x.
    const Index np = 0; ///< The number of variables p.
    const Index ny = 0; ///< The number of variables y.
    const Index nz = 0; ///< The number of variables z.
    const Index nw = 0; ///< The number of variables w = (y, z).

    NewtonStepOptions options; ///< The options for the Newton step calculation.

    NewtonStepSolverRangespace rangespace; ///< The Newton step solver based on a rangespace algorithm.
    NewtonStepSolverNullspace nullspace;   ///< The Newton step solver based on a nullspace algorithm.
    NewtonStepSolverFullspace fullspace;   ///< The Newton step solver based on a fullspace algorithm.

    Vector dx; ///< The auxiliary Newton step vector dx.
    Vector dp; ///< The auxiliary Newton step vector dp.
    Vector dw; ///< The auxiliary Newton step vector dw.

    Vector dwbar; ///< The auxiliary Newton step vector Δw' = (Δw'bs, Δw'bu, Δw'bl).

    Impl(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz),
      rangespace(nx, np, ny, nz),
      nullspace(nx, np, ny, nz),
      fullspace(nx, np, ny, nz),
      dx(zeros(nx)), dp(zeros(np)), dw(zeros(nw)), dwbar(zeros(nw))
    {
    }

    /// Decompose the canonical Jacobian matrix.
    auto decomposeCanonical(const JacobianMatrix& J) -> void
    {
        switch(options.method)
        {
        case NewtonStepMethod::Nullspace: nullspace.decompose(J); break;
        case NewtonStepMethod::Rangespace: rangespace.decompose(J); break;
        default: fullspace.decompose(J); break;
        }
    }

    /// Solve the canonical Newton step problem.
    auto solveCanonical(const JacobianMatrix& J, const ResidualVector& F, CanonicalVectorRef dus) -> void
    {
        switch(options.method)
        {
        case NewtonStepMethod::Nullspace: nullspace.compute(J, F, dus); break;
        case NewtonStepMethod::Rangespace: rangespace.compute(J, F, dus); break;
        default: fullspace.compute(J, F, dus); break;
        }
    }

    auto decompose(const JacobianMatrix& J) -> void
    {
        decomposeCanonical(J);
    }

    auto compute(const JacobianMatrix& J, const ResidualVector& F, SolutionVector& du) -> void
    {
        const auto dims = J.dims();
        const auto Jbar = J.canonicalForm();

        const auto R = Jbar.R;

        const auto js = Jbar.js;
        const auto ju = Jbar.ju;

        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nl  = dims.nl;

        auto dxs = dx.head(ns);

        auto dwbs = dwbar.head(nbs);
        auto dwbu = dwbar.segment(nbs, nbu);
        auto dwbl = dwbar.tail(nl);

        solveCanonical(J, F, {dxs, dp, dwbs});

        dwbu.fill(0.0);
        dwbl.fill(0.0);

        dw.noalias() = tr(R) * dwbar;

        du.x(js) = dxs;
        du.x(ju).fill(0.0);
        du.p = dp;
        du.y = dw.head(ny);
        du.z = dw.tail(nz);
    }
};

NewtonStepSolver::NewtonStepSolver(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
{}

NewtonStepSolver::NewtonStepSolver(const NewtonStepSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

NewtonStepSolver::~NewtonStepSolver()
{}

auto NewtonStepSolver::operator=(NewtonStepSolver other) -> NewtonStepSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto NewtonStepSolver::setOptions(const NewtonStepOptions& options) -> void
{
    pimpl->options = options;
}

auto NewtonStepSolver::options() const -> const NewtonStepOptions&
{
    return pimpl->options;
}

auto NewtonStepSolver::decompose(const JacobianMatrix& M) -> void
{
    pimpl->decompose(M);
}

auto NewtonStepSolver::compute(const JacobianMatrix& J, const ResidualVector& F, SolutionVector& du) -> void
{
    pimpl->compute(J, F, du);
}

} // namespace Optima

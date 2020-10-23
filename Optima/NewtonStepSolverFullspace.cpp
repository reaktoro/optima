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

#include "NewtonStepSolverFullspace.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/JacobianMatrix.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/LU.hpp>

namespace Optima {

struct NewtonStepSolverFullspace::Impl
{
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods
    LU lu;      ///< The LU decomposition solver.
    JacobianMatrix::Dims dims; ///< The dimension details of the Jacobian matrix.

    Impl(Index nx, Index np, Index ny, Index nz)
    : mat(nx + np + ny + nz, nx + np + ny + nz),
      vec(nx + np + ny + nz)
    {
    }

    auto decompose(const JacobianMatrix& J) -> void
    {
        dims = J.dims();

        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;

        const auto t = ns + np + nbs;

        auto M = mat.topLeftCorner(t, t);

        auto M1 = M.topRows(nbs);
        auto M2 = M.middleRows(nbs, nns);
        auto M3 = M.middleRows(nbs + nns, np);
        auto M4 = M.bottomRows(nbs);

        const auto Jbar = J.canonicalForm();

        const auto Hbsbs = Jbar.Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = Jbar.Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = Jbar.Hss.bottomRows(nns).leftCols(nbs);
        const auto Hnsns = Jbar.Hss.bottomRows(nns).rightCols(nns);

        const auto Hbsp = Jbar.Hsp.topRows(nbs);
        const auto Hnsp = Jbar.Hsp.bottomRows(nns);

        const auto Sbsns = Jbar.Sbn.topLeftCorner(nbs, nns);
        const auto Sbsp  = Jbar.Sbp.topRows(nbs);

        const auto Vpbs = Jbar.Vps.leftCols(nbs);
        const auto Vpns = Jbar.Vps.rightCols(nns);
        const auto Vpp = Jbar.Vpp;

        const auto Ibsbs = identity(nbs, nbs);

        const auto Opbs  = zeros(np, nbs);
        const auto Obsbs = zeros(nbs, nbs);

        if(nbs) M1 << Hbsbs, Hbsns, Hbsp, Ibsbs;
        if(nns) M2 << Hnsbs, Hnsns, Hnsp, tr(Sbsns);
        if( np) M3 << Vpbs, Vpns, Vpp, Opbs;
        if(nbs) M4 << Ibsbs, Sbsns, Sbsp, Obsbs;

        lu.decompose(M);
    }

    auto compute(const JacobianMatrix& J, const ResidualVector& F, CanonicalVectorRef dus) -> void
    {
        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;

        const auto t = ns + np + nbs;

        auto r = vec.head(t);

        auto xbs = r.head(nbs);
        auto xns = r.segment(nbs, nns);
        auto p   = r.segment(nbs + nns, np);
        auto wbs = r.tail(nbs);

        const auto Fbar = F.canonicalForm();

        const auto axs  = Fbar.axs;
        const auto ap   = Fbar.ap;
        const auto awbs = Fbar.awbs;

        r << axs, ap, awbs;

        lu.solve(r);

        dus.xs << xbs, xns;
        dus.p = p;
        dus.wbs = wbs;
    }
};

NewtonStepSolverFullspace::NewtonStepSolverFullspace(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
{}

NewtonStepSolverFullspace::NewtonStepSolverFullspace(const NewtonStepSolverFullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

NewtonStepSolverFullspace::~NewtonStepSolverFullspace()
{}

auto NewtonStepSolverFullspace::operator=(NewtonStepSolverFullspace other) -> NewtonStepSolverFullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto NewtonStepSolverFullspace::decompose(const JacobianMatrix& M) -> void
{
    pimpl->decompose(M);
}

auto NewtonStepSolverFullspace::compute(const JacobianMatrix& J, const ResidualVector& F, CanonicalVectorRef dus) -> void
{
    pimpl->compute(J, F, dus);
}

} // namespace Optima

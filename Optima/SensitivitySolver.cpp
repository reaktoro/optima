// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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

#include "SensitivitySolver.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LinearSolver.hpp>

namespace Optima {

struct SensitivitySolver::Impl
{
    MasterDims dims;           ///< The dimensions of the master variables.
    LinearSolver linearsolver; ///< The linear solver for the master matrix equations.
    MasterVector r;            ///< The right-hand side vector in the linear system problems for sensitivity computation.
    Index nc = 0;              ///< The number of sensitivity parameters *c*.
    Matrix bc;                 ///< The Jacobian matrix of *b* with respect to the sensitivity parameters *c*.

    Impl()
    {
        LinearSolverOptions options;
        options.method = LinearSolverMethod::Nullspace;
        linearsolver.setOptions(options);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        dims = problem.dims;
        r.resize(dims);
        nc = problem.c.size();
        bc = problem.bc;
        errorif(nc && bc.cols() != nc, "MasterProblem::bc has ", bc.cols(), " columns but expected is ", nc, ".");
        errorif(nc && bc.rows() != dims.ny, "MasterProblem::bc has ", bc.rows(), " rows but expected is ", dims.ny, ".");
    }

    auto solve(const ResidualFunction& F, const MasterState& state, MasterSensitivity& sensitivity) -> void
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

        auto& [xc, pc, wc, sc] = sensitivity;

        xc.resize(nx, nc);
        pc.resize(np, nc);
        wc.resize(nw, nc);
        sc.resize(nx, nc);

        const auto& res = F.result();
        const auto& fxc = res.f.fxc;
        const auto& hc  = res.h.ddc;
        const auto& vc  = res.v.ddc;

        assert( fxc.rows() == nx );
        assert(  hc.rows() == nz );
        assert(  vc.rows() == np );

        const auto& js  = res.stabilitystatus.js;
        const auto& ju  = res.stabilitystatus.ju;
        const auto& jms = res.stabilitystatus.jms;

        const auto& Jc = res.Jc;

        linearsolver.decompose(Jc);

        for(Index i = 0; i < nc; ++i)
        {
            r.x(js) = -fxc.col(i)(js);
            r.x(ju).fill(0.0);
            r.p = -vc.col(i);
            r.w.topRows(ny) = bc.col(i);
            r.w.bottomRows(nz) = -hc.col(i);
            linearsolver.solve(Jc, r, { xc.col(i), pc.col(i), wc.col(i) });
        }

        xc(jms, all).fill(0.0); // remove derivatives associated with meta-stable basic variables!

        const auto Wx = res.Jm.W.Wx;

        sc(js, all).fill(0.0);
        sc(ju, all) = fxc(ju, all) + tr(Wx(all, ju)) * wc;
    }
};

SensitivitySolver::SensitivitySolver()
: pimpl(new Impl())
{}

SensitivitySolver::SensitivitySolver(const SensitivitySolver& other)
: pimpl(new Impl(*other.pimpl))
{}

SensitivitySolver::~SensitivitySolver()
{}

auto SensitivitySolver::operator=(SensitivitySolver other) -> SensitivitySolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SensitivitySolver::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto SensitivitySolver::solve(const ResidualFunction& F, const MasterState& state, MasterSensitivity& sensitivity) -> void
{
    pimpl->solve(F, state, sensitivity);
}

} // namespace Optima

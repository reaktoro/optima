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
    Matrix bc;                 ///< The Jacobian matrix *∂b/∂c* with respect to sensitive parameters *c*.

    Impl()
    {
        LinearSolverOptions options;
        options.method = LinearSolverMethod::Nullspace;
        linearsolver.setOptions(options);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        dims = problem.dims;
        bc   = problem.bc;
        r.resize(dims);
    }

    auto solve(const ResidualFunction& F, const MasterState& state, MasterSensitivity& sensitivity) -> void
    {
        const auto res = F.result();

        const auto& fxc = res.f.fxc;
        const auto& hc = res.h.ddc;
        const auto& vc = res.v.ddc;

        auto& xc  = sensitivity.xc;
        auto& pc  = sensitivity.pc;
        auto& wc  = sensitivity.wc;
        auto& sc  = sensitivity.sc;

        const auto nx = dims.nx;
        const auto np = dims.np;
        const auto ny = dims.ny;
        const auto nz = dims.nz;
        const auto nw = dims.nw;
        const auto nc = xc.cols();

        assert( fxc.rows() == nx );
        assert(  hc.rows() == nz );
        assert(  vc.rows() == np );
        assert(  bc.rows() == ny );
        assert(  xc.rows() == nx );
        assert(  pc.rows() == np );
        assert(  wc.rows() == nw );
        assert(  sc.rows() == nx );
        assert( fxc.cols() == nc );
        assert(  hc.cols() == nc );
        assert(  bc.cols() == nc );
        assert(  vc.cols() == nc );
        assert(  xc.cols() == nc );
        assert(  pc.cols() == nc );
        assert(  wc.cols() == nc );
        assert(  sc.cols() == nc );

        const auto& js = state.js;
        const auto& ju = state.ju;
        const auto& Jc = res.Jc;

        linearsolver.decompose(Jc);

        for(Index i = 0; i < nw; ++i)
        {
            r.x(js) = -fxc.col(i)(js);
            r.x(ju).fill(0.0);
            r.p = -vc.col(i);
            r.w.topRows(ny) = bc.col(i);
            r.w.bottomRows(nz) = -hc.col(i);

            linearsolver.solve(Jc, r, { xc.col(i), pc.col(i), wc.col(i) });
        }

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

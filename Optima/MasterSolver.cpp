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

#include "MasterSolver.hpp"

// Optima includes
#include <Optima/Convergence.hpp>
#include <Optima/ErrorControl.hpp>
#include <Optima/Exception.hpp>
#include <Optima/MasterProblem.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/NewtonStep.hpp>
#include <Optima/Options.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>
#include <Optima/Result.hpp>
#include <Optima/TransformStep.hpp>

namespace Optima {

const auto CONTINUE = true;
const auto STOP     = false;

struct MasterSolver::Impl
{
    const MasterDims dims;
    ResidualFunction F;
    ResidualErrors E;
    MasterVector uo;
    NewtonStep newtonstep;
    TransformStep transformstep;
    ErrorControl errorcontrol;
    Convergence convergence;
    Result result;
    Options options;

    Impl(const MasterDims& dims, MatrixConstRef Ax, MatrixConstRef Ap)
    : dims(dims), F(dims, Ax, Ap), E(dims), uo(dims.nx, dims.np, dims.nw),
      newtonstep(dims),
      transformstep(dims),
      errorcontrol(dims),
      convergence()
    {
    }

    auto solve(MasterProblem problem, MasterVectorRef u) -> Result
    {
        initialize(problem, u);
        while(stepping())
            step(u);
        finalize();
        return result;
    }

    auto initialize(MasterProblem problem, MasterVectorRef u) -> bool
    {
        result = {};
        F.initialize(problem);
        E.initialize({ problem.xlower, problem.xupper });
        newtonstep.initialize({ problem.xlower, problem.xupper, options.newtonstep });
        transformstep.initialize({ problem.xlower, problem.xupper, problem.phi });
        convergence.initialize({ options.convergence });
    }

    auto stepping() -> bool
    {
        if(result.iterations > options.maxiterations)
            return STOP;

        if(convergence.converged())
            return STOP;

        return CONTINUE;
    }

    auto step(MasterVectorRef u) -> void
    {
        result.iterations += 1;
        F.update(u);
        newtonstep.apply(F, uo, u);
        transformstep.execute(uo, u, F, E);
        errorcontrol.execute(uo, u, F, E);
        convergence.update(E);
    }

    auto finalize() -> void
    {
    }
};

MasterSolver::MasterSolver(const MasterDims& dims, MatrixConstRef Ax, MatrixConstRef Ap)
: pimpl(new Impl(dims, Ax, Ap))
{}

MasterSolver::MasterSolver(const MasterSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

MasterSolver::~MasterSolver()
{}

auto MasterSolver::operator=(MasterSolver other) -> MasterSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto MasterSolver::setOptions(const Options& options) -> void
{
    pimpl->options = options;
}

auto MasterSolver::solve(MasterProblem problem, MasterVectorRef u) -> Result
{
    return pimpl->solve(problem, u);
}

} // namespace Optima

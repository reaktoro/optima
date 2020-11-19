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

    Impl(const MasterDims& dims)
    : dims(dims), F(dims), E(dims), uo(dims),
      newtonstep(dims),
      transformstep(dims),
      errorcontrol(dims),
      convergence()
    {
    }

    auto solve(const MasterProblem& problem, MasterVectorRef u) -> Result
    {
        initialize(problem, u);
        while(stepping())
            step(u);
        finalize();
        return result;
    }

    auto setOptions(const Options& opts) -> bool
    {
        options = opts;
        newtonstep.setOptions(opts.newtonstep);
        convergence.setOptions(opts.convergence);
    }

    auto initialize(const MasterProblem& problem, MasterVectorRef u) -> bool
    {
        result = {};
        uo = u;
        F.initialize(problem);
        E.initialize(problem);
        transformstep.initialize(problem);
        newtonstep.initialize(problem);
        errorcontrol.initialize(problem);
        convergence.initialize(problem);
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
        E.update(u, F); // TODO: This call may not be needed since it may be executed below
        newtonstep.apply(F, uo, u);
        transformstep.execute(uo, u, F, E);
        errorcontrol.execute(uo, u, F, E);
        convergence.update(E);
        uo = u;
    }

    auto finalize() -> void
    {
        result.succeeded = convergence.converged();
    }
};

MasterSolver::MasterSolver(const MasterDims& dims)
: pimpl(new Impl(dims))
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
    pimpl->setOptions(options);
}

auto MasterSolver::solve(const MasterProblem& problem, MasterVectorRef u) -> Result
{
    return pimpl->solve(problem, u);
}

} // namespace Optima

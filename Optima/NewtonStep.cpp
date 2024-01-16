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

#include "NewtonStep.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LinearSolver.hpp>

namespace Optima {

struct NewtonStep::Impl
{
    MasterDims dims;           ///< The dimensions of the master variables.
    LinearSolver linearsolver; ///< The linear solver for the master matrix equations.
    MasterVector du;           ///< The Newton step for master variables u = (x, p, w).
    Vector xlower;             ///< The lower bounds for variables *x*.
    Vector xupper;             ///< The upper bounds for variables *x*.
    Vector plower;             ///< The lower bounds for variables *p*.
    Vector pupper;             ///< The upper bounds for variables *p*.

    Impl()
    {}

    auto setOptions(const NewtonStepOptions options) -> void
    {
        linearsolver.setOptions(options.linearsolver);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        dims = problem.dims;
        xlower = problem.xlower;
        xupper = problem.xupper;
        plower = problem.plower;
        pupper = problem.pupper;
        du.resize(dims);
    }

    auto apply(const ResidualFunction& F, MasterVectorView uo, MasterVectorRef u) -> void
    {
        sanitycheck();
        const auto res = F.result();
        const auto Jc = res.Jc;
        const auto Fc = res.Fc;
        linearsolver.decompose(Jc);
        linearsolver.solve(Jc, Fc, du);
        u.x.noalias() = uo.x + du.x;
        u.p.noalias() = uo.p + du.p;
        u.w.noalias() = uo.w + du.w;
    }

    auto sanitycheck() const -> void
    {
        assert(xlower.size() == dims.nx);
        assert(xupper.size() == dims.nx);
        assert(plower.size() == dims.np);
        assert(pupper.size() == dims.np);
    }
};

NewtonStep::NewtonStep()
: pimpl(new Impl())
{}

NewtonStep::NewtonStep(const NewtonStep& other)
: pimpl(new Impl(*other.pimpl))
{}

NewtonStep::~NewtonStep()
{}

auto NewtonStep::operator=(NewtonStep other) -> NewtonStep&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto NewtonStep::setOptions(const NewtonStepOptions& options) -> void
{
    pimpl->setOptions(options);
}

auto NewtonStep::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto NewtonStep::apply(const ResidualFunction& F, MasterVectorView uo, MasterVectorRef u) -> void
{
    pimpl->apply(F, uo, u);
}

} // namespace Optima

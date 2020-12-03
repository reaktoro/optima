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

#include "ErrorStatus.hpp"

// Optima includes
#include <Optima/Exception.hpp>

namespace Optima {

struct ErrorStatus::Impl
{
    /// The options for error analysis during an optimization calculation.
    ErrorStatusOptions options;

    /// The error of the optimization calculation corresponding to initial guess.
    double errorfirst = 0.0;

    /// The error in the previous iteration of the optimization calculation.
    double errorprev = 0.0;

    /// The current error in the optimization calculation.
    double error = 0.0;

    Impl()
    {
    }

    auto initialize(ErrorStatusInitializeArgs args) -> void
    {
        options = args.options;
        errorfirst = args.errorfirst;
        errorprev = args.errorfirst;
        error = args.errorfirst;
        sanitycheck();
    }

    auto update(const ResidualFunction& F) -> void
    {
        errorprev = error;
        // error = F.residualErrors().error;
        sanitycheck();
    }

    auto errorHasDecreased() const -> bool
    {
        return error < errorprev;
    }

    auto errorHasDecreasedSignificantly() const -> bool
    {
        return error < options.significantly_decreased * errorprev;
    }

    auto errorHasIncreased() const -> bool
    {
        return error > errorprev;
    }

    auto errorHasIncreasedSignificantly() const -> bool
    {
        return error > options.significantly_increased * errorprev;
    }

    auto errorIsntFinite() const -> bool
    {
        return !std::isfinite(error);
    }

    auto sanitycheck() const -> void
    {
        assert(options.significantly_increased > 0.0);
        assert(options.significantly_decreased > 0.0);
        assert(options.significantly_increased_initial > 0.0);
        assert(errorfirst > 0.0);
        assert(errorprev >= 0.0);
        assert(error >= 0.0);
    }
};

ErrorStatus::ErrorStatus()
: pimpl(new Impl())
{}

ErrorStatus::ErrorStatus(const ErrorStatus& other)
: pimpl(new Impl(*other.pimpl))
{}

ErrorStatus::~ErrorStatus()
{}

auto ErrorStatus::operator=(ErrorStatus other) -> ErrorStatus&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ErrorStatus::update(const ResidualFunction& F) -> void
{
    pimpl->update(F);
}

auto ErrorStatus::errorHasDecreased() const -> bool
{
    return pimpl->errorHasDecreased();
}

auto ErrorStatus::errorHasDecreasedSignificantly() const -> bool
{
    return pimpl->errorHasDecreasedSignificantly();
}

auto ErrorStatus::errorHasIncreased() const -> bool
{
    return pimpl->errorHasIncreased();
}

auto ErrorStatus::errorHasIncreasedSignificantly() const -> bool
{
    return pimpl->errorHasIncreasedSignificantly();
}

auto ErrorStatus::errorIsntFinite() const -> bool
{
    return pimpl->errorIsntFinite();
}

auto ErrorStatus::error() const -> double
{
    return pimpl->error;
}

} // namespace Optima

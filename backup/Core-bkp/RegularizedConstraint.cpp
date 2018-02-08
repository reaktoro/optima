// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include "Constraint.hpp"

namespace Optima {

struct Constraint::Impl
{
    /// The original matrix representing the linear constraints.
    Matrix Aorig;

    /// The matrix corresponding to the linearly independent constraints.
    Matrix Ali;

    /// The regularized matrix that equivalently represents the linear constraints.
    Matrix Areg;

    /// The indices of the linearly independent constraints.
    Indices li_constraints;
};

Constraint::Constraint()
{
    
}

Constraint::Constraint(const Matrix& A)
{
    
}

virtual Constraint::~Constraint()
{
    
}

auto Constraint::xlower(Index index, double value) -> void
{
    
}

auto Constraint::xlower(double value) -> void
{
    
}

auto Constraint::xupper(Index index, double value) -> void
{
    
}

auto Constraint::xupper(double value) -> void
{
    
}

auto Constraint::xequal(Index index, double value) -> void
{
    
}

auto Constraint::equality(const Matrix& Ae, const Vector& be) -> void
{
    
}

auto Constraint::equality(const Vector& be) -> void
{
    
}

auto Constraint::inequality(const Matrix& Ai, const Vector& bi) -> void
{
    
}

auto Constraint::inequality(const Vector& bi) -> void
{
    
}

auto Constraint::isrationalAe(bool value) -> void
{
    
}

auto Constraint::isrationalAi(bool value) -> void
{
    
}

auto Constraint::regularize(const Vector& x) -> void
{
    
}

auto Constraint::xlower() const -> const Vector&
{
    
}

auto Constraint::xupper() const -> const Vector&
{
    
}

auto Constraint::Ae() const -> const Matrix&
{
    
}

auto Constraint::Ai() const -> const Matrix&
{
    
}

auto Constraint::be() const -> const Vector&
{
    
}

auto Constraint::bi() const -> const Vector&
{
    
}

auto Constraint::Re() const -> const Matrix&
{
    
}

auto Constraint::Ri() const -> const Matrix&
{
    
}

auto Constraint::Bn() const -> const Matrix&
{
    
}

auto Constraint::b() const -> const Vector&
{
    
}

auto Constraint::ibasic() const -> const Indices&
{
    
}

auto Constraint::inonbasic() const -> const Indices&
{
    
}

auto Constraint::ifixed() const -> const Indices&
{
    
}

} // namespace Optima

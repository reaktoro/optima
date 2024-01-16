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

#pragma once

// Optima includes
#include <Optima/MasterDims.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

template<typename Vec>
struct MasterVectorBase;

using MasterVector     = MasterVectorBase<Vector>;
using MasterVectorRef  = MasterVectorBase<VectorRef>;
using MasterVectorView = MasterVectorBase<VectorView>;

/// Used as a base template type for master vector types.
template<typename Vec>
struct MasterVectorBase
{
    Vec x; ///< The vector *x* in *u = (x, p, w)*.
    Vec p; ///< The vector *p* in *u = (x, p, w)*.
    Vec w; ///< The vector *w* in *u = (x, p, w)*.

    /// Construct a default MasterVectorBase object.
    MasterVectorBase()
    {}

    /// Construct a MasterVectorBase object.
    MasterVectorBase(const MasterDims& dims)
    : MasterVectorBase(dims.nx, dims.np, dims.nw) {}

    /// Construct a MasterVectorBase object.
    MasterVectorBase(Index nx, Index np, Index nw)
    : MasterVectorBase(zeros(nx + np + nw), nx, np, nw) {}

    /// Construct a MasterVectorBase object.
    template<typename Data>
    MasterVectorBase(Data&& data, Index nx, Index np, Index nw)
    : x(data.head(nx)), p(data.segment(nx, np)), w(data.tail(nw)) {}

    /// Construct a MasterVectorBase object.
    MasterVectorBase(const Vec& x, const Vec& p, const Vec& w)
    : x(x), p(p), w(w) {}

    /// Construct a MasterVectorBase object.
    MasterVectorBase(Vec& x, Vec& p, Vec& w)
    : x(x), p(p), w(w) {}

    /// Construct a MasterVectorBase object with given immutable MasterVectorBase object.
    template<typename V>
    MasterVectorBase(const MasterVectorBase<V>& other)
    : x(other.x), p(other.p), w(other.w) {}

    /// Construct a MasterVectorBase object with given mutable MasterVectorBase object.
    template<typename V>
    MasterVectorBase(MasterVectorBase<V>& other)
    : x(other.x), p(other.p), w(other.w) {}

    /// Assign a MasterVectorBase object to this.
    template<typename V>
    auto operator=(const MasterVectorBase<V>& other) -> MasterVectorBase&
    {
        x.noalias() = other.x;
        p.noalias() = other.p;
        w.noalias() = other.w;
        return *this;
    }

    /// Add a MasterVectorBase object to this.
    template<typename V>
    auto operator+=(const MasterVectorBase<V>& other) -> MasterVectorBase&
    {
        x += other.x; p += other.p; w += other.w; return *this;
    }

    /// Subtract a MasterVectorBase object from this.
    template<typename V>
    auto operator-=(const MasterVectorBase<V>& other) -> MasterVectorBase&
    {
        x -= other.x; p -= other.p; w -= other.w; return *this;
    }

    /// Multiply this MasterVectorBase object by a scalar.
    auto operator*=(double s) -> MasterVectorBase&
    {
        x *= s; p *= s; w *= s; return *this;
    }

    /// Divide this MasterVectorBase object by a scalar.
    auto operator/=(double s) -> MasterVectorBase&
    {
        x /= s; p /= s; w /= s; return *this;
    }

    /// Resise this MasterVectorBase object with given dimensions.
    auto resize(const MasterDims& dims) -> void
    {
        x = zeros(dims.nx);
        p = zeros(dims.np);
        w = zeros(dims.nw);
    }

    /// Return the dot product of this MasterVectorBase object with another.
    template<typename V>
    auto dot(const MasterVectorBase<V>& v) const -> double
    {
        return x.dot(v.x) + p.dot(v.p) + w.dot(v.w);
    }

    /// Return the Euclidean norm of this MasterVectorBase object.
    auto norm() const -> double
    {
        return std::sqrt(squaredNorm());
    }

    /// Return the squared Euclidean norm of this MasterVectorBase object.
    auto squaredNorm() const -> double
    {
        return x.squaredNorm() + p.squaredNorm() + w.squaredNorm();
    }

    /// Return the size of this MasterVectorBase object.
    auto size() const { return x.size() + p.size() + w.size(); }

    /// Convert this MasterVectorBase object into a Vector object.
    operator Vector() const { Vector res(size()); res << x, p, w; return res; }
};

template<typename L, typename R>
auto operator+(const MasterVectorBase<L>& l, const MasterVectorBase<R>& r)
{
    return MasterVectorBase{l.x + r.x, l.p + r.p, l.w + r.w};
}

template<typename L, typename R>
auto operator-(const MasterVectorBase<L>& l, const MasterVectorBase<R>& r)
{
    return MasterVectorBase{l.x - r.x, l.p - r.p, l.w - r.w};
}

template<typename V>
auto operator*(double l, const MasterVectorBase<V>& r)
{
    return MasterVectorBase{l * r.x, l * r.p, l * r.w};
}

template<typename V>
auto operator*(const MasterVectorBase<V>& l, double r)
{
    return r * l;
}

template<typename V>
auto operator/(const MasterVectorBase<V>& l, double r)
{
    return (1.0/r) * l;
}

} // namespace Optima

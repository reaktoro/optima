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

#pragma once

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used as a base template type for master vector types.
template<typename Vec>
struct MasterVectorBase
{
    Vec x; ///< The vector *x* in *u = (x, p, w)*.
    Vec p; ///< The vector *p* in *u = (x, p, w)*.
    Vec w; ///< The vector *w* in *u = (x, p, w)*.

    /// Construct a MasterVectorBase object. // TODO: This constructor must accept instead MasterDims!
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

    /// Return the size of this MasterVectorBase object.
    auto size() const { return x.size() + p.size() + w.size(); }

    /// Convert this MasterVectorBase object into a Vector object.
    operator Vector() const { Vector res(size()); res << x, p, w; return res; }
};

using MasterVector     = MasterVectorBase<Vector>;
using MasterVectorRef  = MasterVectorBase<VectorRef>;
using MasterVectorView = MasterVectorBase<VectorConstRef>;

} // namespace Optima

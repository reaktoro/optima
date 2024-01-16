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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used as a base template type for canonical vector types.
template<typename Vec>
struct CanonicalVectorBase
{
    Vec xs;  ///< The vector *xs* in *uc = (xs, xu, p, wbs)*.
    Vec xu;  ///< The vector *xu* in *uc = (xs, xu, p, wbs)*.
    Vec p;   ///< The vector *p* in *uc = (xs, xu, p, wbs)*.
    Vec wbs; ///< The vector *wbs* in *uc = (xs, xu, p, wbs)*.

    /// Construct a CanonicalVectorBase object.
    CanonicalVectorBase(Index ns, Index nu, Index np, Index nbs)
    : CanonicalVectorBase(zeros(ns + nu + np + nbs), ns, nu, np, nbs) {}

    /// Construct a CanonicalVectorBase object.
    template<typename Data>
    CanonicalVectorBase(Data&& data, Index ns, Index nu, Index np, Index nbs)
    : xs(data.head(ns)), xu(data.segment(ns, nu)), p(data.segment(ns + nu, np)), wbs(data.tail(nbs)) {}

    /// Construct a CanonicalVectorBase object.
    CanonicalVectorBase(const Vec& xs, const Vec& xu, const Vec& p, const Vec& wbs)
    : xs(xs), xu(xu), p(p), wbs(wbs) {}

    /// Construct a CanonicalVectorBase object.
    CanonicalVectorBase(Vec& xs, Vec& xu, Vec& p, Vec& wbs)
    : xs(xs), xu(xu), p(p), wbs(wbs) {}

    /// Construct a CanonicalVectorBase object with given immutable CanonicalVectorBase object.
    template<typename V>
    CanonicalVectorBase(const CanonicalVectorBase<V>& other)
    : xs(other.xs), xu(other.xu), p(other.p), wbs(other.wbs) {}

    /// Construct a CanonicalVectorBase object with given mutable CanonicalVectorBase object.
    template<typename V>
    CanonicalVectorBase(CanonicalVectorBase<V>& other)
    : xs(other.xs), xu(other.xu), p(other.p), wbs(other.wbs) {}

    /// Return the size of this CanonicalVectorBase object.
    auto size() const { return xs.size() + xu.size() + p.size() + wbs.size(); }

    /// Convert this CanonicalVectorBase object into a Vector object.
    operator Vector() const { Vector res(size()); res << xs, xu, p, wbs; return res; }
};

using CanonicalVector     = CanonicalVectorBase<Vector>;
using CanonicalVectorRef  = CanonicalVectorBase<VectorRef>;
using CanonicalVectorView = CanonicalVectorBase<VectorView>;

} // namespace Optima

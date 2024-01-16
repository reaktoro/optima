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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/CanonicalDims.hpp>
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/MasterMatrix.hpp>

namespace Optima {

/// Used to assemble the canonical form of a master matrix.
class Canonicalizer
{
public:
    /// Construct a Canonicalizer instance.
    Canonicalizer();

    /// Construct a Canonicalizer instance.
    explicit Canonicalizer(const MasterMatrix& M);

    /// Construct a copy of a Canonicalizer instance.
    Canonicalizer(const Canonicalizer& other);

    /// Destroy this Canonicalizer instance.
    virtual ~Canonicalizer();

    /// Assign a Canonicalizer instance to this.
    auto operator=(Canonicalizer other) -> Canonicalizer&;

    /// Assemble the canonical form of the master matrix.
    auto update(const MasterMatrix& M) -> void;

    /// Return an immutable view to the canonical form of a master matrix.
    auto canonicalMatrix() const -> CanonicalMatrix;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

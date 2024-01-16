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
#include <Optima/MasterDims.hpp>
#include <Optima/MatrixViewRWQ.hpp>
#include <Optima/MatrixViewW.hpp>

namespace Optima {

/// Used to compute the echelon form of matrix *W = [Ax Ap; Jx Jp]*.
class EchelonizerW
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a default EchelonizerW object.
    EchelonizerW();

    /// Construct a copy of a EchelonizerW object.
    EchelonizerW(const EchelonizerW& other);

    /// Destroy this EchelonizerW object.
    virtual ~EchelonizerW();

    /// Assign a EchelonizerW object to this.
    auto operator=(EchelonizerW other) -> EchelonizerW& = delete;

    /// Initialize only once the *Ax* and *Ap* matrices in case these seldom change.
    auto initialize(const MasterDims& dims, MatrixView Ax, MatrixView Ap) -> void;

    /// Update the echelon form of matrix *W* where only *Jx* and *Jp* have changed.
    auto update(MatrixView Jx, MatrixView Jp, VectorView weights) -> void;

    /// Return an immutable view to the assembled matrix *W*.
    auto W() const -> MatrixViewW;

    /// Return an immutable view to the echelon form of *W*.
    auto RWQ() const -> MatrixViewRWQ;
};

} // namespace Optima

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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class MasterMatrixH;
class MasterMatrixV;
class MasterMatrixW;

/// The dimension details of a master matrix in its canonical form.
class CanonicalDims
{
public:
    Index nx;  ///< The number of variables x.
    Index np;  ///< The number of variables p.
    Index ny;  ///< The number of variables y.
    Index nz;  ///< The number of variables z.
    Index nw;  ///< The number of variables w = (y, z).
    Index ns;  ///< The number of stable variables in x.
    Index nu;  ///< The number of unstable variables in x.
    Index nb;  ///< The number of basic variables in x.
    Index nn;  ///< The number of non-basic variables in x.
    Index nl;  ///< The number of linearly dependent rows in Wx = [Ax; Jx].
    Index nbs; ///< The number of stable basic variables.
    Index nbu; ///< The number of unstable basic variables.
    Index nns; ///< The number of stable non-basic variables.
    Index nnu; ///< The number of unstable non-basic variables.
    Index nbe; ///< The number of stable explicit basic variables.
    Index nbi; ///< The number of stable implicit basic variables.
    Index nne; ///< The number of stable explicit non-basic variables.
    Index nni; ///< The number of stable implicit non-basic variables.
};

/// Used to represent a master matrix in its canonical form.
class CanonicalMatrix
{
public:
    CanonicalDims dims;   ///< The dimension details of a master matrix in its canonical form.
    MatrixConstRef Hss;   ///< The matrix Hss in the canonical master matrix.
    MatrixConstRef Hsp;   ///< The matrix Hsp in the canonical master matrix.
    MatrixConstRef Vps;   ///< The matrix Vps in the canonical master matrix.
    MatrixConstRef Vpp;   ///< The matrix Vpp in the canonical master matrix.
    MatrixConstRef Sbsns; ///< The matrix Sbsns in the canonical master matrix.
    MatrixConstRef Sbsp;  ///< The matrix Sbsp in the canonical master matrix.
};

/// Used to export all details of a master matrix in its canonical form.
class CanonicalDetails
{
public:
    CanonicalDims dims;   ///< The dimension details of a master matrix in its canonical form.
    MatrixConstRef Hss;   ///< The matrix Hss in the canonical master matrix.
    MatrixConstRef Hsp;   ///< The matrix Hsp in the canonical master matrix.
    MatrixConstRef Vps;   ///< The matrix Vps in the canonical master matrix.
    MatrixConstRef Vpp;   ///< The matrix Vpp in the canonical master matrix.
    MatrixConstRef Sbn;   ///< The matrix Sbn in the canonical master matrix.
    MatrixConstRef Sbp;   ///< The matrix Sbp in the canonical master matrix.
    MatrixConstRef R;     ///< The echelonizer matrix R so that R*W*Q = [Ibb Sbn Sbp], Q = [jb jn].
    MatrixConstRef Ws;    ///< The matrix Ws in W' = [Ws Wu Wp].
    MatrixConstRef Wu;    ///< The matrix Wu in W' = [Ws Wu Wp].
    MatrixConstRef Wp;    ///< The matrix Wp in W' = [Ws Wu Wp].
    MatrixConstRef As;    ///< The matrix As in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    MatrixConstRef Au;    ///< The matrix Au in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    MatrixConstRef Ap;    ///< The matrix Ap in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    MatrixConstRef Js;    ///< The matrix Js in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    MatrixConstRef Ju;    ///< The matrix Ju in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    MatrixConstRef Jp;    ///< The matrix Jp in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
    IndicesConstRef jb;   ///< The indices of the basic variables ordered as jb = (jbs, jbu).
    IndicesConstRef jn;   ///< The indices of the non-basic variables ordered as jn = (jns, jnu).
    IndicesConstRef js;   ///< The indices of the stable variables ordered as js = (jbs, jns).
    IndicesConstRef ju;   ///< The indices of the unstable variables ordered as ju = (jbu, jnu).
};

/// Used to represent the master matrix.
/// A master matrix is a matrix with the following structure:
/// \eqc{M=\begin{bmatrix}H_{\mathrm{xx}} & H_{\mathrm{xp}} & W_{\mathrm{x}}^{T}\\V_{\mathrm{px}} & V_{\mathrm{pp}} & 0\\W_{\mathrm{x}} & W_{\mathrm{p}} & 0\end{bmatrix}.}
class MasterMatrix
{
public:
    /// Construct a MasterMatrix instance.
    MasterMatrix(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a MasterMatrix instance.
    MasterMatrix(const MasterMatrix& other);

    /// Destroy this MasterMatrix instance.
    virtual ~MasterMatrix();

    /// Assign a MasterMatrix instance to this.
    auto operator=(MasterMatrix other) -> MasterMatrix&;

    /// Update the master matrix with a canonicalization process.
    auto update(const MasterMatrixH& H, const MasterMatrixV& V, const MasterMatrixW& W, IndicesConstRef ju) -> void;

    /// Return the master matrix in its canonical representation.
    auto canonicalMatrix() const -> CanonicalMatrix;

    /// Return a view to the components of the canonical form of the master matrix.
    auto canonicalForm() const -> CanonicalDetails;

    /// Convert this MasterMatrix object into a Matrix object.
    auto matrix() const -> Matrix;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

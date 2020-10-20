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
#include <Optima/JacobianBlockH.hpp>
#include <Optima/JacobianBlockV.hpp>
#include <Optima/JacobianBlockW.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the Jacobian matrix of the optimization problem.
class JacobianMatrix
{
public:
    /// Construct a JacobianMatrix instance.
    JacobianMatrix(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a JacobianMatrix instance.
    JacobianMatrix(const JacobianMatrix& other);

    /// Destroy this JacobianMatrix instance.
    virtual ~JacobianMatrix();

    /// Assign a JacobianMatrix instance to this.
    auto operator=(JacobianMatrix other) -> JacobianMatrix&;

    /// Update the Jacobian matrix.
    auto update(const JacobianBlockH& H, const JacobianBlockV& V, const JacobianBlockW& W, IndicesConstRef ju) -> void;

    /// The dimension details of the Jacobian matrix and its canonical form.
    struct Dims
    {
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

    /// Return a view to the components of the canonical form of the Jacobian matrix.
    auto dims() const -> Dims;

    /// The matrix components in the canonical form of the Jacobian matrix.
    struct CanonicalForm
    {
        MatrixConstRef  Hss;   ///< The matrix Hss in the canonical Jacobian matrix.
        MatrixConstRef  Hsp;   ///< The matrix Hsp in the canonical Jacobian matrix.
        MatrixConstRef  Vps;   ///< The matrix Vps in the canonical Jacobian matrix.
        MatrixConstRef  Vpp;   ///< The matrix Vpp in the canonical Jacobian matrix.
        MatrixConstRef  Sbsns; ///< The matrix Sbsns in the canonical Jacobian matrix.
        MatrixConstRef  Sbsp;  ///< The matrix Sbsp in the canonical Jacobian matrix.
        MatrixConstRef  R;     ///< The canonicalizer matrix R so that R*W*Q = [Ibb Sbn Sbp], Q = [jb jn].
        MatrixConstRef  Ws;    ///< The matrix Ws in W' = [Ws Wu Wp].
        MatrixConstRef  Wu;    ///< The matrix Wu in W' = [Ws Wu Wp].
        MatrixConstRef  Wp;    ///< The matrix Wp in W' = [Ws Wu Wp].
        MatrixConstRef  As;    ///< The matrix As in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        MatrixConstRef  Au;    ///< The matrix Au in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        MatrixConstRef  Ap;    ///< The matrix Ap in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        MatrixConstRef  Js;    ///< The matrix Js in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        MatrixConstRef  Ju;    ///< The matrix Ju in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        MatrixConstRef  Jp;    ///< The matrix Jp in W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].
        IndicesConstRef jb;    ///< The indices of the basic variables ordered as jb = (jbs, jbu).
        IndicesConstRef jn;    ///< The indices of the non-basic variables ordered as jn = (jns, jnu).
        IndicesConstRef js;    ///< The indices of the stable variables ordered as js = (jbs, jns).
        IndicesConstRef ju;    ///< The indices of the unstable variables ordered as ju = (jbu, jnu).
    };

    /// Return a view to the components of the canonical form of the Jacobian matrix.
    auto canonicalForm() const -> CanonicalForm;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

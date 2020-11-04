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
#include <Optima/SaddlePointTypes.hpp>

namespace Optima {

// Forward declarations
class SaddlePointOptions;

/// The arguments for constructor of class SaddlePointSolver.
struct SaddlePointSolverInitArgs
{
    Index nx = 0;        ///< The dimension of vector *x* of the saddle point problem.
    Index np = 0;        ///< The dimension of vector *p* of the saddle point problem.
    Index ny = 0;        ///< The dimension of vector *y* of the saddle point problem.
    Index nz = 0;        ///< The dimension of vector *z* of the saddle point problem.
    MatrixConstRef Ax;   ///< The matrix block *Ax* of the saddle point problem.
    MatrixConstRef Ap;   ///< The matrix block *Ap* of the saddle point problem.
};

/// The arguments for method SaddlePointSolver::canonicalize.
struct SaddlePointSolverCanonicalize1Args
{
    MatrixConstRef  Hxx; ///< The matrix block *Hxx* of the saddle point problem.
    MatrixConstRef  Hxp; ///< The matrix block *Hxp* of the saddle point problem.
    MatrixConstRef  Vpx; ///< The matrix block *Vpx* of the saddle point problem.
    MatrixConstRef  Vpp; ///< The matrix block *Vpp* of the saddle point problem.
    MatrixConstRef  Jx;  ///< The matrix block *Jx* of the saddle point matrix.
    MatrixConstRef  Jp;  ///< The matrix block *Jp* of the saddle point matrix.
    IndicesConstRef ju;  ///< The indices of the unstable variables *xu* in *x = (xs, xu)*.
    VectorConstRef  wx;  ///< The priority weights for variables in *x* to become basic variables.
};

/// The arguments for method SaddlePointSolver::canonicalize.
struct SaddlePointSolverCanonicalize2Args
{
    MatrixConstRef Jx;  ///< The matrix block *Jx* of the saddle point matrix.
    VectorConstRef wx;  ///< The priority weights for variables in *x* to become basic variables.
};

/// The arguments for method SaddlePointSolver::decompose.
struct SaddlePointSolverDecomposeArgs
{
    MatrixConstRef  Hxx; ///< The matrix block *Hxx* of the saddle point problem.
    MatrixConstRef  Hxp; ///< The matrix block *Hxp* of the saddle point problem.
    MatrixConstRef  Vpx; ///< The matrix block *Vpx* of the saddle point problem.
    MatrixConstRef  Vpp; ///< The matrix block *Vpp* of the saddle point problem.
    MatrixConstRef  Jx;  ///< The matrix block *Jx* of the saddle point matrix.
    MatrixConstRef  Jp;  ///< The matrix block *Jp* of the saddle point matrix.
    IndicesConstRef ju;  ///< The indices of the unstable variables *xu* in *x = (xs, xu)*.
};

/// The arguments for method SaddlePointSolver::rhs.
/// Use this method if you are solving the following matrix equation:
/// @eqc{\begin{bmatrix}H_{\mathrm{ss}} & 0 & H_{\mathrm{sp}} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\0 & I_{\mathrm{uu}} & 0 & 0 & 0\\V_{\mathrm{ps}} & 0 & V_{\mathrm{pp}} & 0 & 0\\J_{\mathrm{s}} & 0 & J_{\mathrm{p}} & 0 & 0\\A_{\mathrm{s}} & A_{\mathrm{u}} & A_{\mathrm{p}} & 0 & 0\end{bmatrix}\begin{bmatrix}x_{\mathrm{s}}\\x_{\mathrm{u}}\\p\\z\\y\end{bmatrix}=\begin{bmatrix}a_{\mathrm{s}}\\a_{\mathrm{u}}\\a_{\mathrm{p}}\\a_{\mathrm{z}}\\a_{\mathrm{y}}\end{bmatrix}}
struct SaddlePointSolverRhs1Args
{
    VectorConstRef ax;   ///< The right-hand side vector *ax* of the saddle point problem.
    VectorConstRef ap;   ///< The right-hand side vector *ap* of the saddle point problem.
    VectorConstRef ay;   ///< The right-hand side vector *ay* of the saddle point problem.
    VectorConstRef az;   ///< The right-hand side vector *az* of the saddle point problem.
};

/// The arguments for method SaddlePointSolver::rhs.
/// Use this method if you are solving the following matrix equation:
/// @eqc{\begin{bmatrix}H_{\mathrm{ss}} & 0 & H_{\mathrm{sp}} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\[1mm]0 & I_{\mathrm{uu}} & 0 & 0 & 0\\[1mm]V_{\mathrm{ps}} & 0 & V_{\mathrm{pp}} & 0 & 0\\[1mm]J_{\mathrm{s}} & 0 & J_{\mathrm{p}} & 0 & 0\\[1mm]A_{\mathrm{s}} & A_{\mathrm{u}} & A_{\mathrm{p}} & 0 & 0\end{bmatrix}\begin{bmatrix}\Delta x_{\mathrm{s}}\\[1mm]\Delta x_{\mathrm{u}}\\[1mm]\Delta p\\[1mm]\Delta z\\[1mm]\Delta y\end{bmatrix}=-\begin{bmatrix}g_{\mathrm{s}}+A_{\mathrm{s}}^{T}\bar{y}+J_{\mathrm{s}}^{T}\bar{z}\\[1mm]0\\[1mm]v\\[1mm]h\\[1mm]A_{\mathrm{s}}\bar{x}_{\mathrm{s}}+A_{\mathrm{u}}\bar{x}_{\mathrm{u}}+A_{\mathrm{p}}\bar{p}-b\end{bmatrix}}
struct SaddlePointSolverRhs2Args
{
    VectorConstRef fx;   ///< The right-hand side vector *fx* of the saddle point problem.
    VectorConstRef x;    ///< The right-hand side vector *x* of the saddle point problem.
    VectorConstRef p;    ///< The right-hand side vector *p* of the saddle point problem.
    VectorConstRef y;    ///< The right-hand side vector *y* of the saddle point problem.
    VectorConstRef z;    ///< The right-hand side vector *z* of the saddle point problem.
    VectorConstRef v;    ///< The right-hand side vector *v* of the saddle point problem.
    VectorConstRef h;    ///< The right-hand side vector *h* of the saddle point problem.
    VectorConstRef b;    ///< The right-hand side vector *b* of the saddle point problem.
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolve1Args
{
    VectorRef sx;         ///< The solution vector *sx* of the saddle point problem.
    VectorRef sp;         ///< The solution vector *sp* of the saddle point problem.
    VectorRef sy;         ///< The solution vector *sy* of the saddle point problem.
    VectorRef sz;         ///< The solution vector *sz* of the saddle point problem.
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolve2Args
{
    VectorConstRef ax;    ///< The right-hand side vector *ax* of the saddle point problem.
    VectorConstRef ap;    ///< The right-hand side vector *ap* of the saddle point problem.
    VectorConstRef ay;    ///< The right-hand side vector *ay* of the saddle point problem.
    VectorConstRef az;    ///< The right-hand side vector *az* of the saddle point problem.
    VectorRef sx;         ///< The solution vector *sx* of the saddle point problem.
    VectorRef sp;         ///< The solution vector *sp* of the saddle point problem.
    VectorRef sy;         ///< The solution vector *sy* of the saddle point problem.
    VectorRef sz;         ///< The solution vector *sz* of the saddle point problem.
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolve3Args
{
    VectorConstRef asu;   ///< The right-hand side vector *(as, au)* of the canonical saddle point problem.
    VectorConstRef ap;    ///< The right-hand side vector *ap* of the canonical saddle point problem.
    VectorConstRef awbs;  ///< The right-hand side vector *awbs* of the canonical saddle point problem.
    VectorRef sx;         ///< The solution vector *sx* of the canonical saddle point problem.
    VectorRef sp;         ///< The solution vector *sp* of the canonical saddle point problem.
    VectorRef sy;         ///< The solution vector *sy* of the canonical saddle point problem.
    VectorRef sz;         ///< The solution vector *sz* of the canonical saddle point problem.
};

/// The arguments for method SaddlePointSolver::multiply.
/// Use method SaddlePointSolver::multiply to compute the matrix-vector multplication below:
/// @eqc{\begin{bmatrix}H_{\mathrm{ss}} & 0 & H_{\mathrm{sp}} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\0 & I_{\mathrm{uu}} & 0 & 0 & 0\\V_{\mathrm{ps}} & 0 & V_{\mathrm{pp}} & 0 & 0\\J_{\mathrm{s}} & 0 & J_{\mathrm{p}} & 0 & 0\\A_{\mathrm{s}} & 0 & A_{\mathrm{p}} & 0 & 0\end{bmatrix}\begin{bmatrix}r_{\mathrm{s}}\\r_{\mathrm{u}}\\r_{\mathrm{p}}\\r_{\mathrm{z}}\\r_{\mathrm{y}}\end{bmatrix}=\begin{bmatrix}a_{\mathrm{s}}\\a_{\mathrm{u}}\\a_{\mathrm{p}}\\a_{\mathrm{z}}\\a_{\mathrm{y}}\end{bmatrix}}
struct SaddlePointSolverMultiplyArgs
{
    VectorConstRef rx;    ///< The multplied vector *rx = (rs, ru)*.
    VectorConstRef rp;    ///< The multplied vector *rp*.
    VectorConstRef ry;    ///< The multplied vector *ry*.
    VectorConstRef rz;    ///< The multplied vector *rz*.
    VectorRef ax;         ///< The computed right-hand size vector *ax = (as, au)*.
    VectorRef ap;         ///< The computed right-hand size vector *ap*.
    VectorRef ay;         ///< The computed right-hand size vector *ay*.
    VectorRef az;         ///< The computed right-hand size vector *az*.
};

/// The arguments for method SaddlePointSolver::transposeMultiply.
/// Use method SaddlePointSolver::transposeMultiply to compute the transpose-matrix-vector multplication below:
/// @eqc{\begin{bmatrix}H_{\mathrm{ss}}^{T} & 0 & V_{\mathrm{ps}}^{T} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\[1mm]0 & I_{\mathrm{uu}} & 0 & 0 & 0\\[1mm]H_{\mathrm{sp}}^{T} & 0 & V_{\mathrm{pp}}^{T} & J_{\mathrm{p}}^{T} & A_{\mathrm{p}}^{T}\\[1mm]J_{\mathrm{s}} & 0 & 0 & 0 & 0\\[1mm]A_{\mathrm{s}} & 0 & 0 & 0 & 0\end{bmatrix}\begin{bmatrix}r_{\mathrm{s}}\\[1mm]r_{\mathrm{u}}\\[1mm]r_{\mathrm{p}}\\[1mm]r_{\mathrm{z}}\\[1mm]r_{\mathrm{y}}\end{bmatrix}=\begin{bmatrix}a_{\mathrm{s}}\\[1mm]a_{\mathrm{u}}\\[1mm]a_{\mathrm{p}}\\[1mm]a_{\mathrm{z}}\\[1mm]a_{\mathrm{y}}\end{bmatrix}}
struct SaddlePointSolverTransposeMultiplyArgs
{
    VectorConstRef rx;    ///< The multplied vector *rx = (rs, ru)*.
    VectorConstRef rp;    ///< The multplied vector *rp*.
    VectorConstRef ry;    ///< The multplied vector *ry*.
    VectorConstRef rz;    ///< The multplied vector *rz*.
    VectorRef ax;         ///< The computed right-hand size vector *ax = (as, au)*.
    VectorRef ap;         ///< The computed right-hand size vector *ap*.
    VectorRef ay;         ///< The computed right-hand size vector *ay*.
    VectorRef az;         ///< The computed right-hand size vector *az*.
};

/// The return type of method SaddlePointSolver::info.
struct SaddlePointSolverState
{
    SaddlePointDims dims;  ///< The dimension variables of the canonical saddle point problem.
    IndicesConstRef js;    ///< The indices of the stable variables in *x* as *xs = (xbs, xns)*.
    IndicesConstRef jbs;   ///< The indices of the stable basic variables in *x*.
    IndicesConstRef jns;   ///< The indices of the stable non-basic variables in *x*.
    IndicesConstRef ju;    ///< The indices of the unstable variables in *x* as *xu = (xbu, xnu)*.
    IndicesConstRef jbu;   ///< The indices of the unstable basic variables in *x*.
    IndicesConstRef jnu;   ///< The indices of the unstable non-basic variables in *x*.
    MatrixConstRef  R;     ///< The echelonizer matrix *R* of *Ax*.
    MatrixConstRef  Hss;   ///< The matrix block *Hss* of the canonical saddle point problem.
    MatrixConstRef  Hsp;   ///< The matrix block *Hsp* of the canonical saddle point problem.
    MatrixConstRef  Vps;   ///< The matrix block *Vps* of the canonical saddle point problem.
    MatrixConstRef  Vpp;   ///< The matrix block *Vpp* of the canonical saddle point problem.
    MatrixConstRef  As;    ///< The matrix block *As* of the canonical saddle point problem.
    MatrixConstRef  Au;    ///< The matrix block *Js* of the canonical saddle point problem.
    MatrixConstRef  Ap;    ///< The matrix block *Jp* of the canonical saddle point problem.
    MatrixConstRef  Js;    ///< The matrix block *Js* of the canonical saddle point problem.
    MatrixConstRef  Jp;    ///< The matrix block *Jp* of the canonical saddle point problem.
    MatrixConstRef  Sbsns; ///< The matrix block *Sbsns* of the canonical saddle point problem.
    MatrixConstRef  Sbsp;  ///< The matrix block *Sbsp* of the canonical saddle point problem.
    VectorConstRef  as;    ///< The assembled right-hand side vector *as* of the canonical saddle point problem.
    VectorConstRef  au;    ///< The assembled right-hand side vector *au* of the canonical saddle point problem.
    VectorConstRef  ap;    ///< The assembled right-hand side vector *ap* of the canonical saddle point problem.
    VectorConstRef  ay;    ///< The assembled right-hand side vector *ay* of the canonical saddle point problem.
    VectorConstRef  az;    ///< The assembled right-hand side vector *az* of the canonical saddle point problem.
    VectorConstRef  aw;    ///< The assembled right-hand side vector *aw* of the canonical saddle point problem.
};

/// Used to solve saddle point problems.
/// Use this class to solve saddle point problems.
///
/// @note There is no need for matrix \eq{A} to have linearly independent rows.
/// The algorithm is able to ignore the linearly dependent rows automatically.
/// However, it is expected that vector \eq{b} in the saddle point matrix have
/// consistent values when linearly dependent rows in \eq{A} exists.
/// For example, assume \eq{Ax = b} represents:
/// \eqc{
/// \begin{bmatrix}1 & 1 & 1 & 1\\0 & 1 & 1 & 1\\1 & 0 & 0 & 0\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\\x_{3}\\x_{4}\end{bmatrix}=\begin{bmatrix}b_{1}\\b_{2}\\b_{3}\end{bmatrix}.
/// }
/// Note that the third row of \eq{A} is linearly dependent on the other two
/// rows: \eq{\text{row}_3=\text{row}_1-\text{row}_2}.
/// Thus, it is expected that an input for vector \eq{b} is consistent with
/// the dependence relationship \eq{b_{3}=b_{1}-b_{2}}.
class SaddlePointSolver
{
public:
    /// Construct a SaddlePointSolver instance with given data.
    SaddlePointSolver(SaddlePointSolverInitArgs args);

    /// Construct a copy of a SaddlePointSolver instance.
    SaddlePointSolver(const SaddlePointSolver& other);

    /// Destroy this SaddlePointSolver instance.
    virtual ~SaddlePointSolver();

    /// Assign a SaddlePointSolver instance to this.
    auto operator=(SaddlePointSolver other) -> SaddlePointSolver&;

    /// Set the options for the solution of saddle point problems.
    auto setOptions(const SaddlePointOptions& options) -> void;

    /// Return the current saddle point options.
    auto options() const -> const SaddlePointOptions&;

    /// Canonicalize the coefficient matrix in the saddle point problem.
    auto canonicalize(SaddlePointSolverCanonicalize1Args args) -> void;

    /// Canonicalize the coefficient matrix in the saddle point problem.
    auto canonicalize(SaddlePointSolverCanonicalize2Args args) -> void;

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose() -> void;

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose(SaddlePointSolverDecomposeArgs args) -> void;

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs1Args args) -> void;

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs2Args args) -> void;

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose and @ref rhs have been called before this method.
    auto solve(SaddlePointSolverSolve1Args args) -> void;

    /// Multiply the saddle point matrix with a given vector.
    /// This method computes the following matrix-vector multplication:
    /// @eqc{\begin{bmatrix}H_{\mathrm{ss}} & 0 & H_{\mathrm{sp}} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\0 & I_{\mathrm{uu}} & 0 & 0 & 0\\V_{\mathrm{ps}} & 0 & V_{\mathrm{pp}} & 0 & 0\\J_{\mathrm{s}} & 0 & J_{\mathrm{p}} & 0 & 0\\A_{\mathrm{s}} & 0 & A_{\mathrm{p}} & 0 & 0\end{bmatrix}\begin{bmatrix}r_{\mathrm{s}}\\r_{\mathrm{u}}\\r_{\mathrm{p}}\\r_{\mathrm{z}}\\r_{\mathrm{y}}\end{bmatrix}=\begin{bmatrix}a_{\mathrm{s}}\\a_{\mathrm{u}}\\a_{\mathrm{p}}\\a_{\mathrm{z}}\\a_{\mathrm{y}}\end{bmatrix}}
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto multiply(SaddlePointSolverMultiplyArgs args) -> void;

    /// Multiply the transpose of the saddle point matrix with a given vector.
    /// This method computes the following matrix-vector multplication:
    /// @eqc{\begin{bmatrix}H_{\mathrm{ss}}^{T} & 0 & V_{\mathrm{ps}}^{T} & J_{\mathrm{s}}^{T} & A_{\mathrm{s}}^{T}\\[1mm]0 & I_{\mathrm{uu}} & 0 & 0 & 0\\[1mm]H_{\mathrm{sp}}^{T} & 0 & V_{\mathrm{pp}}^{T} & J_{\mathrm{p}}^{T} & A_{\mathrm{p}}^{T}\\[1mm]J_{\mathrm{s}} & 0 & 0 & 0 & 0\\[1mm]A_{\mathrm{s}} & 0 & 0 & 0 & 0\end{bmatrix}\begin{bmatrix}r_{\mathrm{s}}\\[1mm]r_{\mathrm{u}}\\[1mm]r_{\mathrm{p}}\\[1mm]r_{\mathrm{z}}\\[1mm]r_{\mathrm{y}}\end{bmatrix}=\begin{bmatrix}a_{\mathrm{s}}\\[1mm]a_{\mathrm{u}}\\[1mm]a_{\mathrm{p}}\\[1mm]a_{\mathrm{z}}\\[1mm]a_{\mathrm{y}}\end{bmatrix}}
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto transposeMultiply(SaddlePointSolverTransposeMultiplyArgs args) -> void;

    /// Return the state of the canonical saddle point solver.
    auto state() const -> SaddlePointSolverState;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima

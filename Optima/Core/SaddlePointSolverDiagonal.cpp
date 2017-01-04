//// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
////
//// Copyright (C) 2014-2017 Allan Leal
////
//// This program is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License
//// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//#include "SaddlePointSolver.hpp"
//
//// Optima includes
//#include <Optima/Common/Exception.hpp>
//#include <Optima/Core/SaddlePointMatrix.hpp>
//#include <Optima/Math/Canonicalizer.hpp>
//#include <Optima/Math/EigenExtern.hpp>
//using namespace Eigen;
//
//namespace Optima {
//
//using VectorBlock = decltype(VectorXd().segment(0,0));
//using MatrixBlock = decltype(MatrixXd().block(0,0,0,0));
//
//struct CanonicalSaddlePointMatrixDiagonal
//{
//    /// The basic and non-basic partition of the Hessian matrix in diagonal form.
//    VectorBlock Hb, Hn;
//
//    /// The non-basic and fixed partition of the canonical matrix S.
//    MatrixBlock Sn, Sf;
//};
//
//struct CanonicalSaddlePointMatrixDense
//{
//    /// The basic and non-basic partition of the Hessian matrix in dense form.
//    MatrixBlock Hbb, Hnn, Hnb, Hbn;
//
//    /// The non-basic and fixed partition of the canonical matrix S.
//    MatrixBlock Sn, Sf;
//};
//
//struct CanonicalSaddlePointVector
//{
//    /// The basic, non-basic, and fixed partition of the right-hand side vector *a*.
//    VectorBlock ab, an, af;
//
//    /// The basic partition of the right-hand side vector *b*.
//    VectorBlock bb;
//};
//
//struct CanonicalSaddlePointSolution
//{
//    /// The basic, non-basic, and fixed partition of the solution vector *x*.
//    VectorBlock xb, xn, xf;
//
//    /// The basic partition of the solution vector *y*.
//    VectorBlock yb;
//};
//
//struct SaddlePointSolver::Impl
//{
//    /// The canonicalizer of the Jacobian matrix `A`.
//    Canonicalizer canonicalizer;
//
//    /// The number of rows and columns in A
//    Index m, n;
//
//    /// The number of basic, non-basic, and fixed variables.
//    Index nb, nn, nf;
//
//    /// The priority weights for the selection of basic variables.
//    VectorXd w;
//
//    /// Auxiliary matrices used in the decompose and solve methods.
//    MatrixXd B, T, M;
//
//    /// Auxiliary vectors used in the decompose and solve methods.
//    VectorXd H, a, b;
//
//    /// The basic and non-basic partition of the Hessian matrix in diagonal form.
//    VectorXd Hb, Hn;
//
//    /// The basic and non-basic partition of the Hessian matrix in dense form.
//    MatrixXd Hbb, Hnn, Hnb, Hbn;
//
//    /// The LU solver aplied to the matrix Mb of dimension nb-by-nb to calculate Lb and Ub.
//    Eigen::PartialPivLU<MatrixXd> lu;
//
//    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
//    auto canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
//    {
//        // The result of this method call
//        SaddlePointResult res;
//
//        // Set the number of rows and columns in A
//        m = lhs.A.rows();
//        n = lhs.A.cols();
//
//        // Allocate auxiliary memory
//        H.resize(n);
//        B.resize(m, n);
//        T.resize(n, m);
//        M.resize(m, m);
//        a.resize(n);
//        b.resize(m);
//
//        // Compute the canonical form of matrix A
//        canonicalizer.compute(lhs.A);
//
//        return res.stop();
//    }
//
//    /// Update the canonical form of the coefficient matrix \eq{A} of the saddle point problem.
//    auto update(const SaddlePointMatrix& lhs) -> void
//    {
//        // Update the priority weights for the update of the canonical form
//        w.noalias() = inv(lhs.H);
//
//        // Update the canonical form and the ordering of the variables
//        canonicalizer.update(w, lhs.fixed);
//
//        // Update the number of fixed, basic, and non-basic variables
//        nf = lhs.fixed.size();
//        nb = canonicalizer.rows();
//        nn = n - nb - nf;
//    }
//
//    /// Decompose the coefficient matrix of the saddle point problem.
//    auto decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
//    {
//        // The result of this method call
//        SaddlePointResult res;
//
//        // Update the canonical form of the coefficient matrix A
//        update(lhs);
//
//        // Alias to the matrices of the canonicalization process
//        const auto& Q = canonicalizer.Q().indices();
//        const auto& S = canonicalizer.S();
//
//        // Create auxiliary matrix views
//        auto Hb = H.head(nb);
//        auto Hn = H.segment(nb, nn);
//        auto Bn = B.topLeftCorner(nb, nn);
//        auto Tb = T.topLeftCorner(nn, nb);
//        auto Mb = M.topLeftCorner(nb, nb);
//        auto Sn = S.leftCols(nn);
//
//        // Set `H` as the diagonal Hessian according to current canonical ordering
//        H.noalias() = rows(lhs.H, Q);
//
//        // Compute the auxiliary matrices Bb and Bn
//        Bn.noalias() = Sn * diag(inv(Hn));
//        Tb.noalias() = tr(Sn) * diag(Hb);
//
//        // Compute the matrix Mb
//        Mb.noalias()  = Bn * Tb;
//        Mb.noalias() += identity(nb, nb);
//
//        // Compute the LU decomposition of `Mb`.
//        lu.compute(Mb);
//
//        return res.stop();
//    }
//
//    auto solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult
//    {
//        // The result of this method call
//        SaddlePointResult res;
//
//        // Alias to members of the saddle point vector solution.
//        auto& x = sol.x;
//        auto& y = sol.y;
//
//        // Resize solution vectors x and y if needed
//        x.resize(n);
//        y.resize(m);
//
//        // Alias to the matrices of the canonicalization process
//        const auto& Q = canonicalizer.Q().indices();
//        const auto& R = canonicalizer.R();
//        const auto& S = canonicalizer.S();
//
//        // Create auxiliary sub-matrix views
//        auto Hb = H.head(nb);
//        auto Hn = H.segment(nb, nn);
//        auto Bn = B.topLeftCorner(nb, nn);
//        auto Tb = T.topLeftCorner(nn, nb);
//        auto Sn = S.leftCols(nn);
//        auto Sf = S.rightCols(nf);
//
//        // Create auxiliary sub-vector views
//        auto ab = a.head(nb);
//        auto an = a.segment(nb, nn);
//        auto af = a.tail(nf);
//        auto bb = b.head(nb);
//        auto yb = y.head(nb);
//
//        // Reorder vector a in the canonical order
//        a.noalias() = rows(rhs.x, Q);
//
//        // Apply the regularizer matrix to b
//        bb.noalias() = R*rhs.y;
//
//        // Compute the saddle point problem solution
//        yb.noalias()  = ab;
//        an.noalias() -= tr(Sn)*yb;
//        bb.noalias() -= Sf*af;
//        bb.noalias() -= Bn*an;
//        ab.noalias()  = lu.solve(bb);
//        an.noalias() += Tb*ab;
//        an.noalias()  = diag(inv(Hn))*an;
//        bb.noalias()  = yb;
//        bb.noalias() -= diag(Hb)*ab;
//
//        // Compute the y vector without canonicalization
//        y.noalias() = tr(R)*bb;
//
//        // Permute back the variables x to their original ordering
//        rows(x, Q).noalias() = a;
//
//        return res.stop();
//    }
//};
//
//SaddlePointSolver::SaddlePointSolver()
//: pimpl(new Impl())
//{}
//
//SaddlePointSolver::SaddlePointSolver(const SaddlePointSolver& other)
//: pimpl(new Impl(*other.pimpl))
//{}
//
//SaddlePointSolver::~SaddlePointSolver()
//{}
//
//auto SaddlePointSolver::operator=(SaddlePointSolver other) -> SaddlePointSolver&
//{
//    pimpl = std::move(other.pimpl);
//    return *this;
//}
//
//auto SaddlePointSolver::canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
//{
//    return pimpl->canonicalize(lhs);
//}
//
//auto SaddlePointSolver::decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult
//{
//    return pimpl->decompose(lhs);
//}
//
//auto SaddlePointSolver::solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult
//{
//    return pimpl->solve(rhs, sol);
//}
//
//} // namespace Optima

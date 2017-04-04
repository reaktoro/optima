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
//#include "OptimumStepper.hpp"
//
//
//
//// Eigenx includes
//#include <Eigenx/LU.hpp> // todo check if necessary later
//
//
//
//
//// Optima includes
//#include <Optima/Core/OptimumOptions.hpp>
//#include <Optima/Core/OptimumStructure.hpp>
//#include <Optima/Core/OptimumParams.hpp>
//#include <Optima/Core/OptimumState.hpp>
//#include <Optima/Core/SaddlePointMatrix.hpp>
//#include <Optima/Core/SaddlePointResult.hpp>
//#include <Optima/Core/SaddlePointSolver.hpp>
//using namespace Eigen;
//
//namespace Optima {
//
//struct OptimumStepper::Impl
//{
//    /// The structure of the optimization problem.
//    OptimumStructure structure;
//
//    /// The options for the optimization calculation
//    OptimumOptions options;
//
//    /// The solution vector `sol = [dx dy dz dw]`.
//    VectorXd solution;
//
//    /// The right-hand side residual vector `res = [rx ry rz rw]`.
//    VectorXd residual;
//
//    /// The `H` matrix in the KKT equation.
//    MatrixXd H;
//
//    /// The `G` matrix in the KKT equation.
//    MatrixXd G;
//
//    /// The KKT solver.
//    SaddlePointSolver kkt;
//
//    /// The order of the variables as `x = [x(stable) x(unstable) x(fixed)]`.
//    VectorXi iordering;
//
//    /// The number of variables.
//    Index n;
//
//    /// The current number of stable, unstable, free, and fixed variables.
//    Index ns, nu, nx, nf;
//
//    /// The number of equality constraints.
//    Index m;
//
//    /// The total number of variables (x, y, z, w).
//    Index t;
//
//    /// Initialize the stepper with the structure of the optimization problem.
//    auto initialize(const OptimumStructure& strct) -> void
//    {
//        // Set the structure member with the given one
//        structure = strct;
//
//        // Initialize the saddle point solver
//        kkt.canonicalize(structure.A);
//
//        // Initialize the members related to number of variables and constraints
//        n  = structure.n;
//        ns = n;
//        nu = 0;
//        nx = n;
//        nf = 0;
//        m  = structure.A.rows();
//        t  = 2*n + m;
//
//        // Allocate memory for some members
//        H = zeros(n, n);
//        G = zeros(m, m);
//        residual.resize(t);
//        solution.resize(t);
//
//        // Initialize the ordering of the variables
//        iordering.setLinSpaced(n, 0, n);
//    }
//
//    /// Decompose the KKT matrix equation used to compute the step vectors.
//    auto decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//    {
//        // Aliases to some variables
//        const auto& A = structure.A;
//        const auto& x = state.x;
//        const auto& z = state.z;
//
//        // Update the number of fixed and free variables
//        nf = params.ifixed.size();
//        nx = n - nf;
//
//        // Partition the variables into stable and unstable sets
////        auto pred = [&](Index i) { return z[i] < 1.0 && x[i] > options.mu; }; // todo maybe it is important to add condition on x < mu too
////        auto pred = [&](Index i) { return z[i] <= x[i] || x[i] > std::sqrt(options.mu) || z[i] < std::sqrt(options.mu); }; // todo maybe it is important to add condition on x < mu too
////        auto pred = [&](Index i) { return x[i] > std::sqrt(options.mu) || z[i] < std::sqrt(options.mu); }; // todo maybe it is important to add condition on x < mu too
////        auto pred = [&](Index i) { return true; };
//
//        auto pred = [&](Index i) { return x[i] > z[i]; };
//        auto it = std::partition(iordering.data(), iordering.data() + nx, pred);
//
//        // Update the number of stable and unstable variables
//        ns = it - iordering.data();
//        nu = nx - ns;
//
//        // Ensure the number of stable variables are positive
//        assert(ns > 0 && "Could not compute the step."
//            "The number of stable variables must be positive.");
//
//        auto ivx = iordering.head(nx);
//        auto ivf = iordering.tail(nf);
//        auto ivs = ivx.head(ns);
//        auto ivu = ivx.tail(nu);
//
//        // Assemble the matrix H in the KKT equation
//        if(f.hessian.size())
//            H.noalias() = f.hessian;
//        else H.fill(0.0);
//
//        auto ifixed = iordering.tail(nu + nf);
//
//        // Calculate Hss' = Hss + inv(Xs)*Zs
//        for(Index s : ivs) H(s, s) += z[s]/x[s];
//
//        // Calculate Huu' = Huu + inv(Xu)*Zu
//        for(Index u : ivu) H(u, u) += z[u]/x[u];
//
//        for(Index u : ivu) H(u, u) += z[u]/x[u];
//
//
//        G = Au * Huu * tr(Au);
//
//        // Update the decomposition of the KKT matrix
//        kkt.decompose({H, A, G, ifixed});
//
////        // The indices of the stable variables
////        auto istable = iordering.head(ns);
////
////        // The indices of the fixed variables including unstable variables
////        auto ifixed = iordering.tail(nu + nf);
////
////        // Assemble the matrix H in the KKT equation
////        if(f.hessian.size())
////            H.noalias() = f.hessian;
////        else H.fill(0.0);
////
////        // Add the inv(Z)*X corresponding to stable variables
////        for(Index s : istable)
////            H(s, s) += z[s] / x[s];
////
////        // Update the decomposition of the KKT matrix
////        kkt.decompose({H, A, ifixed});
//    }
//
//    /// Solve the KKT matrix equation.
//    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//    {
//        const auto& A  = structure.A;
//
//        auto& x = state.x;
//        auto& y = state.y;
//        auto& z = state.z;
//
//        auto a = residual.head(n);
//        auto b = residual.segment(n, m);
//        auto c = residual.tail(n);
//
//        auto dx = solution.head(n);
//        auto dy = solution.segment(n, m);
//        auto dz = solution.tail(n);
//
//        a.noalias() = -(f.grad - tr(A)*y - z);
//        b.noalias() = -(A*x - params.a);
//        c.noalias() = -(x % z - options.mu);
//
//        // The indices of the stable and unstable variables
//        auto ivs = iordering.head(ns);
//        auto ivu = iordering.segment(ns, nu);
//        auto ivf = iordering.tail(nf);
//
////        auto hessian = [&](Index i, Index j)
////        {
////            return f.hessian.size() ? f.hessian(i,j) : 0.0;
////        };
//
////        for(Index u : ivu) a[u] -= hessian(u,u)*c[u]/z[u];
////        for(Index f : ivf) a[f] = params.xfixed[f];
//
//        for(Index s : ivs) a[s] += c[s]/x[s];
//        for(Index u : ivu) a[u] -= c[u]/x[u];
//        rows(dz, ivu) = rows(a, ivu);
//        rows(a, ivu).fill(0.0);
//        rows(a, ivf) = params.xfixed;
//
//        kkt.solve({a, b}, {dx, dy});
//
//        dy *= -1.0;
//
//        for(Index s : ivs)
//            dz[s] = (c[s] - z[s]*dx[s])/x[s];
//
//        for(Index u : ivu)
//            dz[u] = -(dz[u] + dot(A.col(u), dy))*z[u]/(H(u, u) * x[u]);
//
//        for(Index u : ivu)
//            dx[u] = (c[u] - x[u]*dz[u])/z[u];
//
//
//
////        for(Index s : ivs)
////            dz[s] = (c[s] - z[s]*dx[s])/x[s];
////
////        for(Index u : ivu)
////            dz[u] = -z[u]*(dx[u]/x[u]);
////
////        for(Index u : ivu)
////            dx[u] = (c[u] - x[u]*dz[u])/z[u];
////
////        for(Index s : ivs)
////            dz[s] = (c[s] - z[s]*dx[s])/x[s];
////
////        for(Index u : ivu)
////            dz[u] = -z[u]*dx[u]/x[u];
////
////        for(Index u : ivu)
////            dx[u] = (c[u] - x[u]*dz[u])/z[u];
//
////        // The indices of the stable and unstable variables
////        auto ivs = iordering.head(ns);
////        auto ivu = iordering.segment(ns, nu);
////        auto ivf = iordering.tail(nf);
////
////        for(Index s : ivs) a[s] += c[s]/x[s];          // as = as + inv(Xs)*cs
////        for(Index u : ivu) std::swap(a[u], dz[u] = 0); // au = 0
////        for(Index f : ivf) a[f] = params.xfixed[f];    // af = xf
////
////        kkt.solve({a, b}, {dx, dy});
////
////        dy *= -1.0;
////
////        for(Index s : ivs)
////            dz[s] = (c[s] - z[s]*dx[s])/x[s];
////
////        for(Index u : ivu)
////            dz[u] = -(dz[u] - hessian(u,u)*c[u]/z[u] + dot(A.col(u), dy))/(1 + hessian(u,u)*x[u]/z[u]);
////
////        for(Index u : ivu)
////            dx[u] = (c[u] - x[u]*dz[u])/z[u];
//
//        VectorXd residual_tmp = residual;
//        VectorXd solution_tmp = solution;
//
//        solve2(params, state, f);
//
//        std::cout << "stable  = " << tr(ivs) << std::endl;
//        std::cout << "x       = " << tr(state.x) << std::endl;
//        std::cout << "dx(rs)  = " << tr(solution_tmp.head(n)) << std::endl;
//        std::cout << "dx(lu)  = " << tr(dx) << std::endl;
//        std::cout << "res(dx) = " << tr(abs(solution_tmp.head(n) - dx)) << std::endl;
//        std::cout << std::endl;
//        std::cout << "y       = " << tr(state.y) << std::endl;
//        std::cout << "dy(rs)  = " << tr(solution_tmp.segment(n, m)) << std::endl;
//        std::cout << "dy(lu)  = " << tr(dy) << std::endl;
//        std::cout << "res(dy) = " << tr(abs(solution_tmp.segment(n, m) - dy)) << std::endl;
//        std::cout << std::endl;
//        std::cout << "z       = " << tr(state.z) << std::endl;
//        std::cout << "dz(rs)  = " << tr(solution_tmp.tail(n)) << std::endl;
//        std::cout << "dz(lu)  = " << tr(dz) << std::endl;
//        std::cout << "res(dz) = " << tr(abs(solution_tmp.tail(n) - dz)) << std::endl;
//
//        residual = residual_tmp;
//        solution = solution_tmp;
//    }
//
//    /// Solve the KKT matrix equation.
//    auto solve2(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//    {
//        const auto& x = state.x;
//        const auto& y = state.y;
//        const auto& z = state.z;
//        const auto& A = structure.A;
//        const auto& a = params.a;
//
////        MatrixXd M = zeros(n+m, n+m);
////        if(f.hessian.size())
////            M.topLeftCorner(n, n) = f.hessian;
////        M.topLeftCorner(n, n).diagonal() += z/x;
////        M.topRows(n).rightCols(m) = -tr(A);
////        M.bottomRows(m).leftCols(n) = A;
//
//        MatrixXd M = zeros(t, t);
//        if(f.hessian.size())
//            M.topLeftCorner(n, n) = f.hessian;
//        M.topRows(n).middleCols(n, m) = -tr(A);
//        M.topRightCorner(n, n).diagonal().fill(-1.0);
//        M.middleRows(n, m).leftCols(n) = A;
//        M.bottomLeftCorner(n, n).diagonal() = z;
//        M.bottomRightCorner(n, n).diagonal() = x;
//
//        residual.head(n) = -(f.grad - tr(A)*y - z);
//        residual.segment(n, m) = -(A*x - a);
//        residual.tail(n) = -(x % z - options.mu);
//
//        solution = M.fullPivLu().solve(residual);
//    }
//};
//
//OptimumStepper::OptimumStepper()
//: pimpl(new Impl())
//{}
//
//OptimumStepper::OptimumStepper(const OptimumStepper& other)
//: pimpl(new Impl(*other.pimpl))
//{}
//
//OptimumStepper::~OptimumStepper()
//{}
//
//auto OptimumStepper::operator=(OptimumStepper other) -> OptimumStepper&
//{
//    pimpl = std::move(other.pimpl);
//    return *this;
//}
//
//auto OptimumStepper::setOptions(const OptimumOptions& options) -> void
//{
//    pimpl->options = options;
//    pimpl->kkt.setOptions(options.kkt);
//}
//
//auto OptimumStepper::initialize(const OptimumStructure& structure) -> void
//{
//    pimpl->initialize(structure);
//}
//
//auto OptimumStepper::decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//{
//    pimpl->decompose(params, state, f);
//}
//
//auto OptimumStepper::solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//{
//    pimpl->solve(params, state, f);
//}
//
//auto OptimumStepper::solve2(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//{
//    pimpl->solve2(params, state, f);
//}
//
//auto OptimumStepper::step() const -> VectorXdConstRef
//{
//    return pimpl->solution;
//}
//
//auto OptimumStepper::dx() const -> VectorXdConstRef
//{
//    return pimpl->solution.head(pimpl->n);
//}
//
//auto OptimumStepper::dy() const -> VectorXdConstRef
//{
//    return pimpl->solution.segment(pimpl->n, pimpl->m);
//}
//
//auto OptimumStepper::dz() const -> VectorXdConstRef
//{
//    return pimpl->solution.tail(pimpl->n);
//}
//
//auto OptimumStepper::dw() const -> VectorXdConstRef
//{
//    return pimpl->solution.tail(pimpl->n);
//}
//
//auto OptimumStepper::residual() const -> VectorXdConstRef
//{
//    return pimpl->residual;
//}
//
//auto OptimumStepper::residualOptimality() const -> VectorXdConstRef
//{
//    return pimpl->residual.head(pimpl->n);
//}
//
//auto OptimumStepper::residualFeasibility() const -> VectorXdConstRef
//{
//    return pimpl->residual.segment(pimpl->n, pimpl->m);
//}
//
//auto OptimumStepper::residualComplementarityLowerBounds() const -> VectorXdConstRef
//{
//    return pimpl->residual.tail(pimpl->n);
//}
//
//auto OptimumStepper::residualComplementarityUpperBounds() const -> VectorXdConstRef
//{
//    return pimpl->residual.tail(pimpl->n);
//}
//
////auto OptimumStepper::residualComplementarityInequality() const -> VectorXdConstRef
////{
////
////}
////
////auto OptimumStepper::lhs() const -> MatrixXdConstRef
////{
////
////}
//
//auto OptimumStepper::ifree() const -> VectorXiConstRef
//{
//    return pimpl->iordering.head(pimpl->nx);
//}
//
//auto OptimumStepper::ifixed() const -> VectorXiConstRef
//{
//    return pimpl->iordering.tail(pimpl->nf);
//}
//
//auto OptimumStepper::istable() const -> VectorXiConstRef
//{
//    return pimpl->iordering.head(pimpl->ns);
//}
//
//auto OptimumStepper::iunstable() const -> VectorXiConstRef
//{
//    return pimpl->iordering.segment(pimpl->ns, pimpl->nu);
//}
//
//} // namespace Optima

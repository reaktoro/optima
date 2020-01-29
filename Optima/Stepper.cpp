// // Optima is a C++ library for solving linear and non-linear constrained optimization problems
// //
// // Copyright (C) 2014-2018 Allan Leal
// //
// // This program is free software: you can redistribute it and/or modify
// // it under the terms of the GNU General Public License as published by
// // the Free Software Foundation, either version 3 of the License, or
// // (at your option) any later version.
// //
// // This program is distributed in the hope that it will be useful,
// // but WITHOUT ANY WARRANTY; without even the implied warranty of
// // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// // GNU General Public License for more details.
// //
// // You should have received a copy of the GNU General Public License
// // along with this program. If not, see <http://www.gnu.org/licenses/>.

// #include "Stepper.hpp"

// // Optima includes
// #include <Optima/Exception.hpp>
// #include <Optima/IpSaddlePointMatrix.hpp>
// #include <Optima/IpSaddlePointSolver.hpp>
// #include <Optima/Options.hpp>

// namespace Optima {

// struct Stepper::Impl
// {
//     /// The options for the optimization calculation
//     Options options;

//     /// The solution vector `s = [dx dy dz dw]`.
//     Vector s;

//     /// The right-hand side residual vector `r = [rx ry rz rw]`.
//     Vector r;

//     /// The matrices Z, W, L, U assuming the ordering x = [x(free) x(fixed)].
//     Vector Z, W, L, U;

//     /// The number of variables.
//     Index n;

//     /// The number of free and fixed variables.
//     Index nx, nf;

//     /// The number of equality constraints.
//     Index m;

//     /// The total number of variables (x, y, z, w).
//     Index t;

//     /// The interior-point saddle point solver.
//     IpSaddlePointSolver solver;

//     /// The boolean flag that indices if the solver has been initialized
//     bool initialized = false;

//     /// Construct a default Stepper::Impl instance.
//     Impl()
//     {}

//     /// Initialize the stepper with the structure of the optimization problem.
//     auto initialize(const StepperProblem& problem) -> void
//     {
//         // Update the initialized status of the solver
//         initialized = true;

//         // Auxiliary references
//         const auto H = problem.H;
//         const auto A = problem.A;
//         const auto J = problem.J;
//         const auto ifixed = problem.ifixed;

//         // Initialize the members related to number of variables and constraints
//         n  = H.rows();
//         m  = A.rows() + J.rows();
//         t  = 3*n + m;

//         // Initialize the number of fixed and free variables
//         nf = ifixed.rows();
//         nx = n - nf;

//         // Initialize Z and W with zeros (the dafault value for variables
//         // with fixed values or no lower/upper bounds).
//         Z = zeros(n);
//         W = zeros(n);

//         // Initialize L and U with ones (the dafault value for variables
//         // with fixed values or no lower/upper bounds).
//         L = ones(n);
//         U = ones(n);

//         // Initialize r and s with zeros.
//         r = zeros(t);
//         s = zeros(t);
//     }

//     /// Decompose the interior-point saddle point matrix for diagonal Hessian matrices.
//     auto decompose(const StepperProblem& problem) -> void
//     {
//         // Initialize the solver if not yet
//         if(!initialized)
//             initialize(problem);

//         // Auxiliary references
//         const auto x = problem.x;
//         const auto z = problem.z;
//         const auto w = problem.w;
//         const auto H = problem.H;
//         const auto A = problem.A;
//         const auto J = problem.J;
//         const auto xlower = problem.xlower;
//         const auto xupper = problem.xupper;
//         const auto ilower = problem.ilower;
//         const auto iupper = problem.iupper;
//         const auto ifixed = problem.ifixed;

//         // Update Z and L for the variables with lower bounds
//         Z(ilower) = z(ilower);
// 		L(ilower) = x(ilower) - xlower;

//         // Update W and U for the variables with upper bounds
//         W(iupper) = w(iupper);
//         U(iupper) = x(iupper) - xupper;

//         // Ensure entries in L are positive in case x[i] == lowerbound[i]
// 		for(Index i : ilower) L[i] = L[i] > 0.0 ? L[i] : options.mu;

//         // Ensure entries in U are negative in case x[i] == upperbound[i]
// 		for(Index i : iupper) U[i] = U[i] < 0.0 ? U[i] : -options.mu;

//         // Define the interior-point saddle point matrix
//         IpSaddlePointMatrix spm(H, A, J, Z, W, L, U, ifixed);

//         // Decompose the interior-point saddle point matrix
//         solver.decompose(spm);
//     }

//     /// Solve the interior-point saddle point matrix.
//     auto solve(const StepperProblem& problem) -> void
//     {
//         // Auxiliary references
//         const auto x = problem.x;
//         const auto y = problem.y;
//         const auto z = problem.z;
//         const auto w = problem.w;
//         const auto g = problem.g;
//         const auto h = problem.h;
//         const auto A = problem.A;
//         const auto J = problem.J;
//         const auto ilower = problem.ilower;
//         const auto iupper = problem.iupper;
//         const auto ifixed = problem.ifixed;

//         const auto yA = y.head(A.rows());
//         const auto yJ = y.tail(J.rows());

//         // Views to the sub-vectors in r = [a b c d]
//         auto a = r.head(n);
//         auto b = r.segment(n, m);
//         auto c = r.segment(n + m, n);
//         auto d = r.tail(n);

//         auto bA = b.head(A.rows());
//         auto bJ = b.tail(J.rows());

//         // Calculate the optimality residual vector a
//         a.noalias() = -(g + tr(A) * yA + tr(J) * yJ - z - w);

//         // Set a to zero for fixed variables
//         a(ifixed).fill(0.0);

//         // Calculate the feasibility residual vector b
//         bA.noalias() = -(A*x - problem.b);
//         bJ.noalias() = -h;

//         // Calculate the centrality residual vectors c and d
//         for(Index i : ilower) c[i] = options.mu - L[i] * z[i]; // TODO Check if mu is still needed. Maybe this algorithm no longer needs perturbation.
//         for(Index i : iupper) d[i] = options.mu - U[i] * w[i];

// //        c.fill(0.0); // TODO For example, there is no mu here and this seems to work
// //        d.fill(0.0);

//         // The right-hand side vector of the interior-point saddle point problem
//         IpSaddlePointVector rhs(r, n, m);

//         // The solution vector of the interior-point saddle point problem
//         IpSaddlePointSolution sol(s, n, m);

//         // Solve the saddle point problem
//         solver.solve(rhs, sol);
//     }

//     /// Return the calculated Newton step vector.
//     auto step() const -> IpSaddlePointVector
//     {
//         return IpSaddlePointVector(s, n, m);
//     }

//     /// Return the calculated residual vector for the current optimum state.
//     auto residual() const -> IpSaddlePointVector
//     {
//         return IpSaddlePointVector(r, n, m);
//     }

//     /// Return the assembled interior-point saddle point matrix.
//     auto matrix(const StepperProblem& problem) -> IpSaddlePointMatrix
//     {
//         // Auxiliary references
//         const auto H = problem.H;
//         const auto A = problem.A;
//         const auto J = problem.J;
//         const auto ifixed = problem.ifixed;
//         return IpSaddlePointMatrix(H, A, J, Z, W, L, U, ifixed);
//     }
// };

// Stepper::Stepper()
// : pimpl(new Impl())
// {}

// Stepper::Stepper(const Stepper& other)
// : pimpl(new Impl(*other.pimpl))
// {}

// Stepper::~Stepper()
// {}

// auto Stepper::operator=(Stepper other) -> Stepper&
// {
//     pimpl = std::move(other.pimpl);
//     return *this;
// }

// auto Stepper::setOptions(const Options& options) -> void
// {
//     pimpl->options = options;
//     pimpl->solver.setOptions(options.kkt);
// }

// auto Stepper::decompose(const StepperProblem& problem) -> void
// {
//     return pimpl->decompose(problem);
// }

// auto Stepper::solve(const StepperProblem& problem) -> void
// {
//     return pimpl->solve(problem);
// }

// auto Stepper::matrix(const StepperProblem& problem) -> IpSaddlePointMatrix
// {
//     return pimpl->matrix(problem);
// }

// auto Stepper::step() const -> IpSaddlePointVector
// {
//     return pimpl->step();
// }

// auto Stepper::residual() const -> IpSaddlePointVector
// {
//     return pimpl->residual();
// }

// } // namespace Optima

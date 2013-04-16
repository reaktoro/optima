/*
 * Demo2.cpp
 *
 *  Created on: 15 Apr 2013
 *      Author: allan
 */

// C++ includes
#include <iostream>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// Optima includes
#include <Optima.hpp>
using namespace Optima;

/// The pressure of the system (in units of bar)
const double P = 1.0;

/// The temperature of the system (in units of K)
const double T = 298.15;

/// The universal gas constant (in units of kJ/(mol:K))
const double R = 8.3144621e-3;

/// The number of chemical species in the system
const unsigned num_species = 7;

/// The number of chemical elements in the system
const unsigned num_elements = 3;

/// The chemical species in the system
const std::vector<std::string> species = {"H2O(a)", "H+", "OH-", "HCO3-", "CO2(a)", "CO2(g)", "H2O(g)"};

unsigned iH2Oa = 0;
unsigned iH    = 1;
unsigned iOH   = 2;
unsigned iHCO3 = 3;
unsigned iCO2a = 4;
unsigned iCO2g = 5;
unsigned iH2Og = 6;

/// The standard chemical potential of the species at 298.15K (in units of kJ)
const std::vector<double> potential = {-237.141, 0.0, -157.262, -586.940, -385.974, -394.400, -228.570};

/// The mapping from the species index to the index of its phase
const std::vector<unsigned> species2phase = {0, 0, 0, 0, 0, 1, 1};

/// The mapping from the phase index to the indices of its species
const std::vector<std::vector<unsigned>> phase2species = {{0, 1, 2, 3, 4}, {5, 6}};

/// The formula matrix of the chemical species
const std::vector<std::vector<double>> formula_matrix =
{
    {2, 1, 1, 1, 0, 0, 2}, // chemical element: H
    {1, 0, 1, 3, 2, 2, 1}, // chemical element: O
    {0, 0, 0, 1, 1, 1, 0}  // chemical element: C
};

/// Calculates the total number of moles in a phase
double CalculateMolesPhase(unsigned iphase, const VectorXd& n)
{
    double nsum = 0.0;
    for(unsigned i : phase2species[iphase])
        nsum += n[i];
    return nsum;
}

/// Calculates the molar fraction of a species
double CalculateMolarFraction(unsigned ispecies, const VectorXd& n)
{
    const unsigned iphase = species2phase[ispecies];
    return n[ispecies]/CalculateMolesPhase(iphase, n);
}

/// Calculates the molar fractions of all species
VectorXd CalculateMolarFractions(const VectorXd& n)
{
    VectorXd x(num_species);
    for(unsigned i = 0; i < num_species; ++i)
        x[i] = CalculateMolarFraction(i, n);
    return x;
}

/// Calculates the vector of the chemical potentials of the species
VectorXd CalculateChemicalPotentials(const VectorXd& n)
{
    const Map<const VectorXd> mu0(potential.data(), num_species);

    VectorXd a = CalculateMolarFractions(n);

    a[5] *= P;
    a[6] *= P;

    VectorXd lna = a.array().log();

    return mu0/(R*T) + lna;
}

/// Calculates the Gibbs free energy of the system
double CalculateGibbs(const VectorXd& n)
{
    VectorXd mu = CalculateChemicalPotentials(n);

    return mu.dot(n);
}

/// Calculates the gradient of the Gibbs free energy of the system
VectorXd CalculateGibbsGradient(const VectorXd& n)
{
    return CalculateChemicalPotentials(n);
}

/// Calculates the Hessian of the Gibbs free energy of the system
MatrixXd CalculateGibbsHessian(const VectorXd& n)
{
    MatrixXd hessian = MatrixXd::Zero(num_species, num_species);

    hessian.block(0, 0, 5, 5) = n.segment(0, 5).asDiagonal().inverse();
    hessian.block(0, 0, 5, 5) -= MatrixXd::Ones(5, 5)/n.segment(0, 5).sum();

    hessian.block(5, 5, 2, 2) = n.segment(5, 2).asDiagonal().inverse();
    hessian.block(5, 5, 2, 2) -= MatrixXd::Ones(2, 2)/n.segment(5, 2).sum();

    return hessian;
}

/// Assembles the formula matrix of the system
MatrixXd FormulaMatrix()
{
    MatrixXd W(3, 7);
    for(int i = 0; i < W.rows(); ++i)
        for(int j = 0; j < W.cols(); ++j)
            W(i, j) = formula_matrix[i][j];
    return W;
}

/// Calculates the mass-balance residual
VectorXd CalculateMassBalanceConstraint(const VectorXd& n, const VectorXd& b)
{
    const MatrixXd W = FormulaMatrix();

    return W*n - b;
}

/// Calculates the gradient of the mass-balance constraint function
MatrixXd CalculateMassBalanceConstraintGradient(const VectorXd& n)
{
    return FormulaMatrix();
}

/// Creates the objective function for the chemical equilibrium problem
ObjectiveFunction CreateGibbsObjectiveFunction(double scale = 1.0)
{
    ObjectiveFunction objective = [=](const VectorXd& n) -> ObjectiveResult
    {
        ObjectiveResult f;
        f.func    = scale * CalculateGibbs(n);
        f.grad    = scale * CalculateGibbsGradient(n);
        f.hessian = scale * CalculateGibbsHessian(n);
        return f;
    };

    return objective;
}

/// Creates the mass-balace constraint function for the chemical equilibrium problem
ConstraintFunction CreateGibbsConstraintFunction(const VectorXd& b)
{
    ConstraintFunction constraint = [=](const VectorXd& x) -> ConstraintResult
    {
        ConstraintResult h;
        h.func = CalculateMassBalanceConstraint(x, b);
        h.grad = CalculateMassBalanceConstraintGradient(x);
        return h;
    };

    return constraint;
}

int main()
{
    std::vector<double> nCO2vals = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2};

    double nH2O = 55;

    while(true) {
    std::cout << "Enter the psi scheme (0:Objective, 1:Lagrange, 2:GradLagrange): " << std::endl;
    unsigned psi; std::cin >> psi;

    IPFilterSolver::Options options;
    options.output.active    = false;
    options.output.precision = 10;
    options.output.width     = 20;
    options.max_iter         = 1000;
    options.psi              = (psi == 2) ? GradLagrange : (psi == 1) ? Lagrange : Objective;

    IPFilterSolver::Params params;

    std::cout << "Use safe-step approach when needed? " << std::endl;
    std::cin >> params.safe_step;

    std::cout << "Use default sigma parameters: " << std::endl;
    bool default_sigma; std::cin >> default_sigma;
    if(not default_sigma)
    {
        params.sigma_fast     = 0.5;
        params.sigma_slow     = 0.5;
        params.sigma_safe_max = 0.5;
        params.sigma_safe_min = 0.5;
    }

    std::cout << "Enter initial guess option (0:n[CO2(a)] = nCO2, 1: n[CO2(g)] = nCO2): " << std::endl;
    unsigned nCO2guess; std::cin >> nCO2guess;

    std::vector<IPFilterSolver::Result> results;

    for(double nCO2 : nCO2vals)
    {
        VectorXd b(num_elements);
        b << 2*nH2O, nH2O + 2*nCO2, nCO2;

        OptimumProblem problem;
        problem.SetNumVariables(num_species);
        problem.SetNumConstraints(num_elements);
        problem.SetObjectiveFunction(CreateGibbsObjectiveFunction());
        problem.SetConstraintFunction(CreateGibbsConstraintFunction(b));

        IPFilterSolver solver;

        solver.SetParams(params);
        solver.SetOptions(options);
        solver.SetProblem(problem);

        VectorXd n = VectorXd::Constant(num_species, 1.0e-5);
        n[iH2Oa] = nH2O;
        n[iCO2a] = (nCO2guess == 0) ? nCO2 : n[iCO2a];
        n[iCO2g] = (nCO2guess == 1) ? nCO2 : n[iCO2g];

        try { solver.Solve(n); } catch (...) {}
        //catch(const std::exception& e) { std::cout << e.what() << std::endl; }

        results.push_back(solver.GetResult());
    }

    std::cout << std::left << std::setw(10) << "nCO2";
    std::cout << std::left << std::setw(10) << std::boolalpha << "converged";
    std::cout << std::left << std::setw(10) << "iters";
    std::cout << std::endl;

    for(unsigned i = 0; i < nCO2vals.size(); ++i)
    {
        std::cout << std::left << std::setw(10) << nCO2vals[i];
        std::cout << std::left << std::setw(10) << std::boolalpha << results[i].converged;
        std::cout << std::left << std::setw(10) << results[i].iterations;
        std::cout << std::endl;
    }
    }
}

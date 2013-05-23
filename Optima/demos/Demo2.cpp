/*
 * Demo2.cpp
 *
 *  Created on: 15 Apr 2013
 *      Author: allan
 */

// C++ includes
#include <algorithm>
#include <iostream>

// Eigen includes
#include <Eigen/Dense>
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
const unsigned N = num_species;

/// The number of chemical elements in the system
const unsigned num_elements = 3;
const unsigned E = num_elements;

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
double CalculateMolarFraction(unsigned i, const VectorXd& n)
{
    const unsigned iphase = species2phase[i];
    return n[i]/CalculateMolesPhase(iphase, n);
}

double CalculateMolarFractionGrad(unsigned i, unsigned j, const VectorXd& n)
{
    const unsigned iphase = species2phase[i];
    const unsigned jphase = species2phase[j];

    if(iphase != jphase)
        return 0.0;

    auto kronecker = [](unsigned i, unsigned j) { return (i == j) ? 1.0 : 0.0; };
    const double xi = CalculateMolarFraction(i, n);
    const double nt = CalculateMolesPhase(iphase, n);

    return (kronecker(i, j) - xi)/nt;
}

/// Calculates the molar fractions of all species
VectorXd CalculateMolarFractions(const VectorXd& n)
{
    VectorXd x(num_species);
    for(unsigned i = 0; i < num_species; ++i)
        x[i] = CalculateMolarFraction(i, n);
    return x;
}

MatrixXd CalculateMolarFractionsGrad(const VectorXd& n)
{
    MatrixXd dxdn(num_species, num_species);
    for(unsigned i = 0; i < num_species; ++i)
        for(unsigned j = 0; j < num_species; ++j)
            dxdn(i, j) = CalculateMolarFractionGrad(i, j, n);
    return dxdn;
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
    VectorXd x = CalculateMolarFractions(n);
    MatrixXd dxdn = CalculateMolarFractionsGrad(n);

    MatrixXd U = x.asDiagonal().inverse() * dxdn;
    MatrixXd A = U + U.transpose() - U.transpose() * n.asDiagonal() * U;

    return A;
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
ObjectiveFunction CreateGibbsObjectiveFunction()
{
    ObjectiveFunction objective = [=](const VectorXd& n) -> ObjectiveResult
    {
        ObjectiveResult f;
        f.func    = CalculateGibbs(n);
        f.grad    = CalculateGibbsGradient(n);
        f.hessian = CalculateGibbsHessian(n);

        return f;
    };

    return objective;
}

/// Creates the mass-balace constraint function for the chemical equilibrium problem
ConstraintFunction CreateGibbsConstraintFunction(const VectorXd& b)
{
    ConstraintFunction constraint = [=](const VectorXd& n) -> ConstraintResult
    {
        ConstraintResult h;
        h.func = CalculateMassBalanceConstraint(n, b);
        h.grad = CalculateMassBalanceConstraintGradient(n);

        return h;
    };

    return constraint;
}

int main()
{
    std::vector<double> nCO2vals = {0.1, 0.2, 0.21, 0.22, 0.23, 0.24, 0.3, 0.4, 1.8, 1.84, 1.848, 1.849, 2.0, 3.0, 4.0};
//    std::vector<double> nCO2vals = {1.8, 2.0, 1.8};

    double nH2O = 55;

    IPFilterParams params;

    IPFilterOptions options;
    options.output.active    = true;
    options.output.precision = 8;
    options.output.width     = 15;
    options.max_iterations   = 200;
    options.tolerance1       = 1.0e-6;
    options.tolerance2       = 1.0e-12;
    options.output.scaled    = false;

    std::vector<IPFilterResult> results;

    IPFilterSolver solver;
    solver.SetOptions(options);

    bool first = true;

    VectorXd n, y, z;

    for(double nCO2 : nCO2vals)
    {
        VectorXd b(E);
        b << 2*nH2O, nH2O + 2*nCO2, nCO2;

        if(first)
        {
            n = VectorXd::Constant(N, 1.0e-7);
            n[iH2Oa] = nH2O;
            n[iCO2g] = nCO2;
        }

        OptimumProblem problem;
        problem.SetNumVariables(N);
        problem.SetNumConstraints(E);
        problem.SetObjectiveFunction(CreateGibbsObjectiveFunction());
        problem.SetConstraintFunction(CreateGibbsConstraintFunction(b));

        Optima::Scaling scaling;
        scaling.SetScalingVariables(n);

        solver.SetProblem(problem);

        std::cout << std::scientific << std::setprecision(6);

        std::string bar(105, '=');
        std::cout << bar << std::endl;
        std::cout << "nCO2 = " << nCO2 << std::endl;


        if(not first) solver.SetScaling(scaling);
        solver.Solve(n, y, z);

        first = false;

        results.push_back(solver.GetResult());
    }

    std::cout << std::left << std::setw(15) << "nCO2";
    std::cout << std::left << std::setw(15) << std::boolalpha << "converged";
    std::cout << std::left << std::setw(15) << "iters";
    std::cout << std::left << std::setw(15) << "error";
    std::cout << std::endl;
    std::cout << std::fixed;
    unsigned total = 0;
    for(unsigned i = 0; i < nCO2vals.size(); ++i)
    {
        total += results[i].num_iterations;
        std::cout << std::left << std::setw(15) << nCO2vals[i];
        std::cout << std::left << std::setw(15) << std::boolalpha << results[i].converged;
        std::cout << std::left << std::setw(15) << results[i].num_iterations;
        std::cout << std::left << results[i].error;
        std::cout << std::endl;
    }

    std::cout << "total: " << total << std::endl;
}

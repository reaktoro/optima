/*
 * Demo2.cpp
 *
 *  Created on: 15 Apr 2013
 *      Author: allan
 */

// C++ includes
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
ObjectiveFunction CreateGibbsObjectiveFunction(const VectorXd& D)
{
    ObjectiveFunction objective = [=](const VectorXd& nscaled) -> ObjectiveResult
    {
        VectorXd n = D.asDiagonal() * nscaled;

        ObjectiveResult f;
        f.func    = CalculateGibbs(n);
        f.grad    = CalculateGibbsGradient(n);
        f.hessian = CalculateGibbsHessian(n);

        f.grad    = D.asDiagonal() * f.grad;
        f.hessian = D.asDiagonal() * f.hessian * D.asDiagonal();

        return f;
    };

    return objective;
}

/// Creates the mass-balace constraint function for the chemical equilibrium problem
ConstraintFunction CreateGibbsConstraintFunction(const VectorXd& b, const VectorXd& D)
{
    ConstraintFunction constraint = [=](const VectorXd& nscaled) -> ConstraintResult
    {
        VectorXd n = D.asDiagonal() * nscaled;

        ConstraintResult h;
        h.func = CalculateMassBalanceConstraint(n, b);
        h.grad = CalculateMassBalanceConstraintGradient(n);

        h.grad = h.grad * D.asDiagonal();

        return h;
    };

    return constraint;
}

int main()
{
//    std::vector<double> nCO2vals = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 1.01, 10, 20, 100};
    std::vector<double> nCO2vals = {0.1, 0.2, 0.3, 0.4, 1, 1.01, 10, 20, 100};

    double nH2O = 55;

//    std::cout << "Enter the psi scheme (0:Objective, 1:Lagrange, 2:GradLagrange): " << std::endl;
//    std::cout << "Enter the sigma scheme (0:Default, 1:LOQO): " << std::endl;
//    std::cout << "Enter initial guess option (0:n[CO2(a)] = nCO2, 1: n[CO2(g)] = nCO2): " << std::endl;
    unsigned psi = 0;// std::cin >> psi;
    unsigned sigma = 0;// std::cin >> sigma;
    unsigned nCO2guess = 1;// std::cin >> nCO2guess;

    IPFilterSolver::Options options;
    options.output.active    = true;
    options.output.precision = 8;
    options.output.width     = 15;
    options.max_iter         = 200;
    options.psi              = (psi == 2) ? GradLagrange : (psi == 1) ? Lagrange : Objective;
    options.sigma            = (sigma == 0) ? SigmaDefault : SigmaLOQO;

    IPFilterSolver::Params params;
    params.xlower               = 1.0e-2;
    params.delta_min            = 1.0e-12;
    params.safe_step            = true;
    params.restoration          = true;
    params.neighbourhood_search = true;

    std::vector<IPFilterSolver::Result> results;
    std::vector<std::string> errors;

    IPFilterSolver solver;
    solver.SetParams(params);
    solver.SetOptions(options);

    bool first = true;

    VectorXd nold, yold, zold;

    MatrixXd W = FormulaMatrix();

    for(double nCO2 : nCO2vals)
    {
        VectorXd b(num_elements);
        b << 2*nH2O, nH2O + 2*nCO2, nCO2;

        if(nCO2 == 10.0)
        {
            std::vector<unsigned> idxa = {5, 6};
            std::vector<unsigned> idxi = {0, 1, 2, 3, 4};

            MatrixXd Wa(E, 2);
            MatrixXd Wi(E, N-2);

            VectorXd na(2);
            VectorXd ni(N-2);

            for(unsigned i = 0; i < Wa.cols(); ++i)
            {
                Wa.col(i) = W.col(idxa[i]);
                na[i] = nold[idxa[i]];
            }

            for(unsigned i = 0; i < Wi.cols(); ++i)
            {
                Wi.col(i) = W.col(idxi[i]);
                ni[i] = nold[idxi[i]];
            }

            VectorXd ba = b - Wi*ni;
            MatrixXd Ma = Wa.transpose()*Wa;
            VectorXd rhs = Wa.transpose()*ba;
            na = Ma.lu().solve(rhs);

            nold[iCO2g] = std::max(params.xlower, na[0]);
            nold[iH2Og] = std::max(params.xlower, na[1]);
        }

        VectorXd D = first ? VectorXd::Ones(num_species) : nold;

        OptimumProblem problem;
        problem.SetNumVariables(num_species);
        problem.SetNumConstraints(num_elements);
        problem.SetObjectiveFunction(CreateGibbsObjectiveFunction(D));
        problem.SetConstraintFunction(CreateGibbsConstraintFunction(b, D));

        solver.SetProblem(problem);

        VectorXd n, y, z;
        if(first)
        {
            n = VectorXd::Constant(num_species, 1.0e-5);
            n[iH2Oa] = nH2O;
            n[iCO2a] = (nCO2guess == 0) ? nCO2 : n[iCO2a];
            n[iCO2g] = (nCO2guess == 1) ? nCO2 : n[iCO2g];
        }
        else
        {
            n = nold;
            y = yold;
            z = zold;
        }

        std::string bar(105, '=');
        std::cout << bar << std::endl;
        std::cout << "nCO2 = " << nCO2 << std::endl;

        try
        {
            if(first)
            {
                VectorXd nscaled = D.asDiagonal().inverse() * n;
                solver.Solve(nscaled);
            }
            else
            {
                VectorXd nscaled = D.asDiagonal().inverse() * n;
                VectorXd yscaled = y;
                VectorXd zscaled = D.asDiagonal() * z;
                solver.Solve(nscaled, yscaled, zscaled);
            }

            errors.push_back("None");
        }
        catch(const std::exception& e) { errors.push_back(e.what()); }

        auto state = solver.GetState();

        VectorXd Lx = state.f.grad + state.h.grad.transpose() * state.y;

        std::cout << bar << std::endl;
        std::cout << std::left << std::setw(15) << "x";
        std::cout << std::left << std::setw(15) << "y";
        std::cout << std::left << std::setw(15) << "z";
        std::cout << std::left << std::setw(15) << "D*x";
        std::cout << std::left << std::setw(15) << "D^-1*z";
        std::cout << std::left << std::setw(15) << "Lx";
        std::cout << std::left << std::setw(15) << "D^-1*Lx";
        std::cout << std::endl;

        for(unsigned i = 0; i < num_species; ++i)
        {
            std::cout << std::left << std::setw(15) << state.x[i];
            if(i < num_elements) std::cout << std::left << std::setw(15) << state.y[i];
            else std::cout << std::left << std::setw(15) << "";
            std::cout << std::left << std::setw(15) << state.z[i];
            std::cout << std::left << std::setw(15) << D[i]*state.x[i];
            std::cout << std::left << std::setw(15) << 1.0/D[i]*state.z[i];
            std::cout << std::left << std::setw(15) << Lx[i];
            std::cout << std::left << std::setw(15) << 1.0/D[i]*Lx[i];
            std::cout << std::endl;
        }

        first = false;

        nold = D.asDiagonal() * solver.GetState().x;
        yold = solver.GetState().y;
        zold = D.asDiagonal().inverse() * solver.GetState().z;

        results.push_back(solver.GetResult());
    }

    std::cout << std::left << std::setw(15) << "nCO2";
    std::cout << std::left << std::setw(15) << std::boolalpha << "converged";
    std::cout << std::left << std::setw(15) << "iters";
    std::cout << std::left << std::setw(15) << "error";
    std::cout << std::endl;

    unsigned total = 0;
    for(unsigned i = 0; i < nCO2vals.size(); ++i)
    {
        total += results[i].iterations;
        std::cout << std::left << std::setw(15) << nCO2vals[i];
        std::cout << std::left << std::setw(15) << std::boolalpha << results[i].converged;
        std::cout << std::left << std::setw(15) << results[i].iterations;
        std::cout << std::left << errors[i];
        std::cout << std::endl;
    }

    std::cout << "total: " << total << std::endl;
}

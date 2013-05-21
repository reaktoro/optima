/*
 * BrentSolver.hpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <functional>
#include <stdexcept>

namespace Optima {

class BrentSolver
{
public:
    typedef std::function<double(double)> Function;

    struct Error : public std::exception
    {
        virtual const char* what() const throw();
    };

    struct Options
    {
        unsigned itmax = 100;

        double tolerance = 1.0e-6;
    };

    struct Result
    {
        double fmin;

        double xmin;

        bool converged;

        operator bool() { return converged; }
    };

    struct Statistics
    {
        unsigned niters;

        unsigned nfunc_evals;
    };

    BrentSolver();

    void SetOptions(const Options& options);

    void SetFunction(const Function& f);

    const Options& GetOptions() const;

    const Function& GetFuction() const;

    const Result& GetResult() const;

    const Statistics& GetStatistics() const;

    void Solve(double a, double b, double c);

private:
    Function f;

    Options options;

    Result result;

    Statistics statistics;
};

} /* namespace Optima */

/*
 * QualitySolver.hpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

class QualitySolver
{
public:

    struct Params
    {
        double mu_min = 1.0e-13;

        double sigma_min = 1.0e-08;

        double sigma_max = 1.0e+03;
    };

    struct Data
    {
        VectorXd x;

        VectorXd y;

        VectorXd z;

        VectorXd dx0;

        VectorXd dy0;

        VectorXd dz0;

        VectorXd dx1;

        VectorXd dy1;

        VectorXd dz1;

        double thh;

        double thl;
    };

    QualitySolver();

    void SetData(const Data& data);

    void SetParams(const Params& params);

    const Data& GetData() const;

    Data& GetData();

    double CalculateQuality(double sigma);

    double CalculateSigma();

private:
    Data data;

    Params params;

    VectorXd dxs;

    VectorXd dys;

    VectorXd dzs;
};

} /* namespace Optima */

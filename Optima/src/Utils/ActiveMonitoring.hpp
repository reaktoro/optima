/*
 * ActiveMonitoring.hpp
 *
 *  Created on: 25 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <vector>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// Optima includes
#include <Utils/Index.hpp>

namespace Optima {

class ActiveMonitoring
{
public:
    struct State;

    ActiveMonitoring();

    void SetThreshold(double threshold);

    void SetLowerBounds(const VectorXd& xL);

    void SetUpperBounds(const VectorXd& xU);

    void AddPartition(const Indices& partition);

    void SetDefaultPartitioning(unsigned num_variables);

    void SetPartitioning(const std::vector<Indices>& partitions);

    double GetThreshold() const;

    const VectorXd& GetLowerBounds() const;

    const VectorXd& GetUpperBounds() const;

    const std::vector<Indices>& GetPartitions() const;

    const State& GetState();

    bool IsPartitioningEmpty() const;

    void Initialise(const VectorXd& x);

    void Update(const VectorXd& x);

public:
    struct State
    {
        /// The indices of the partitions that are active at the lower bounds
        Indices lower_active_partitions;

        /// The indices of the partitions that are active at the upper bounds
        Indices upper_active_partitions;

        /// The indices of the lower active partitions that have continuously departed from the lower bounds
        Indices departing_lower_active_partitions;

        /// The indices of the upper active partitions that have continuously departed from the upper bounds
        Indices departing_upper_active_partitions;

        /// The record of the departure progress of the lower active partitions from the lower bounds
        std::vector<std::vector<double>> progress_lower_active_partitions;

        /// The record of the departure progress of the upper active partitions from the upper bounds
        std::vector<std::vector<double>> progress_upper_active_partitions;
    };

private:
    /// The state of the active partitions at the current monitoring stage
    State state;

    /// The active partitioning of the variables
    std::vector<Indices> partitions;

    /// The lower bound of the variables
    VectorXd xL;

    /// The upper bound of the variables
    VectorXd xU;

    /// The threshold value used to determine which partitions are active at the bounds
    double threshold = 1.0e-8;

    /// The boolean value that indicates if the lower bound is negative infinity
    bool inf_lower_bound = true;

    /// The boolean value that indicates if the upper bound is positive infinity
    bool inf_upper_bound = true;

private:
    Indices LowerActivePartitions(const VectorXd& x) const;

    Indices UpperActivePartitions(const VectorXd& x) const;
};

} /* namespace Optima */

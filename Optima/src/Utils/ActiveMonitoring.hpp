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

    /**
     * Constructs a default @ref ActiveMonitoring instance
     */
    ActiveMonitoring();

    /**
     * Sets the threshold used to determine which partitions are active either at the lower or upper bounds
     */
    void SetThreshold(double threshold);

    /**
     * Sets the lower bounds of the components
     */
    void SetLowerBounds(const VectorXd& xL);

    /**
     * Sets the upper bounds of the components
     */
    void SetUpperBounds(const VectorXd& xU);

    /**
     * Adds a new partition
     *
     * @param partition The indices of the components that form the new partition
     */
    void AddPartition(const Indices& partition);

    /**
     * Sets the default partitioning of the components, where each one is a partition
     *
     * @param num_variables The number of variables
     */
    void SetDefaultPartitioning(unsigned num_variables);

    /**
     * Sets the partitioning of the components
     */
    void SetPartitioning(const std::vector<Indices>& partitions);

    /**
     * Gets the threshold used to determine which partitions are active either at the lower or upper bounds
     */
    double GetThreshold() const;

    /**
     * Gets the vector containing the lower bounds of the components
     */
    const VectorXd& GetLowerBounds() const;

    /**
     * Gets the vector containing the upper bounds of the components
     */
    const VectorXd& GetUpperBounds() const;

    /**
     * Gets the partitioning of the components, with each partition being a set of indices of the components
     */
    const std::vector<Indices>& GetPartitions() const;

    /**
     * Gets the current state of the monitoring
     */
    const State& GetState() const;

    /**
     * Checks if the partitioning of the components is empty
     */
    bool EmptyPartitioning() const;

    /**
     * Initialise the state of the monitoring by detecting those lower and upper active partitions
     *
     * @param x The initial values of the variables
     */
    void Initialise(const VectorXd& x);

    /**
     * Updates the state of the monitoring of lower and upper active partitions
     *
     * @param x The new values of the variables
     */
    void Update(const VectorXd& x);

    /**
     * Determines the indices of the partitions that are active at the lower bound at @c x
     *
     * @param x The values of the components where the active partitions will be determined
     *
     * @return The indices of the partitions that are active at the lower bound at @c x
     */
    Indices DetermineLowerActivePartitions(const VectorXd& x) const;

    /**
     * Determines the indices of the partitions that are active at the upper bound at @c x
     *
     * @param x The values of the components where the active partitions will be determined
     *
     * @return The indices of the partitions that are active at the upper bound at @c x
     */
    Indices DetermineUpperActivePartitions(const VectorXd& x) const;

    /**
     * Determines the indices of the lower active partitions that have continuously departed from the lower bounds
     */
    Indices DetermineDepartingLowerActivePartitions() const;

    /**
     * Determines the indices of the upper active partitions that have continuously departed from the lower bounds
     */
    Indices DetermineDepartingUpperActivePartitions() const;

    Indices DetermineDepartingLowerActiveComponents() const;

    Indices DetermineDepartingUpperActiveComponents() const;

public:
    struct State
    {
        /// The indices of the partitions that are active at the lower bounds
        Indices lower_active_partitions;

        /// The indices of the partitions that are active at the upper bounds
        Indices upper_active_partitions;

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
};

} /* namespace Optima */

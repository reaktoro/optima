/*
 * ActiveMonitoring.cpp
 *
 *  Created on: 25 Apr 2013
 *      Author: allan
 */

#include "ActiveMonitoring.hpp"

// C++ includes
#include <algorithm>
#include <cmath>

// Optima includes
#include <Utils/Macros.hpp>

namespace Optima {

ActiveMonitoring::ActiveMonitoring()
{}

void ActiveMonitoring::SetThreshold(double threshold)
{
    this->threshold = threshold;
}

void ActiveMonitoring::SetLowerBounds(const VectorXd& xL)
{
    this->xL = xL;

    inf_lower_bound = std::isinf(xL.maxCoeff()) == -1;
}

void ActiveMonitoring::SetUpperBounds(const VectorXd& xU)
{
    this->xU = xU;

    inf_upper_bound = std::isinf(xU.minCoeff()) == +1;
}

void ActiveMonitoring::AddPartition(const Indices& partition)
{
    partitions.push_back(partition);
}

void ActiveMonitoring::SetDefaultPartitioning(unsigned num_variables)
{
    partitions.resize(num_variables);
    for(unsigned i = 0; i < num_variables; ++i)
        partitions[i].resize(1, i);
}

void ActiveMonitoring::SetPartitioning(const std::vector<Indices>& partitions)
{
    this->partitions = partitions;
}

const std::vector<Indices>& ActiveMonitoring::GetPartitions() const
{
    return partitions;
}

const ActiveMonitoring::State& ActiveMonitoring::GetState()
{
    const unsigned num_lower_active_partitions = state.lower_active_partitions.size();
    const unsigned num_upper_active_partitions = state.upper_active_partitions.size();

    state.departing_lower_active_partitions.clear();
    state.departing_upper_active_partitions.clear();

    if(num_lower_active_partitions)
    {
        for(unsigned i = 0; i < num_lower_active_partitions; ++i)
        {
            const auto& begin = state.progress_lower_active_partitions[i].begin();
            const auto& end = state.progress_lower_active_partitions[i].end();
            if(std::is_sorted(begin, end))
                state.departing_lower_active_partitions.push_back(state.lower_active_partitions[i]);
        }
    }

    if(num_upper_active_partitions)
    {
        for(unsigned i = 0; i < num_upper_active_partitions; ++i)
        {
            const auto& begin = state.progress_upper_active_partitions[i].begin();
            const auto& end = state.progress_upper_active_partitions[i].end();
            if(std::is_sorted(begin, end))
                state.departing_upper_active_partitions.push_back(state.upper_active_partitions[i]);
        }
    }

    return state;
}

bool ActiveMonitoring::IsPartitioningEmpty() const
{
    return partitions.empty();
}

void ActiveMonitoring::Initialise(const VectorXd& x)
{
    state = State();

    state.lower_active_partitions = LowerActivePartitions(x);
    state.upper_active_partitions = UpperActivePartitions(x);

    const unsigned num_lower = state.lower_active_partitions.size();
    const unsigned num_upper = state.upper_active_partitions.size();

    state.progress_lower_active_partitions.resize(num_lower);
    state.progress_upper_active_partitions.resize(num_upper);

    Update(x);
}

void ActiveMonitoring::Update(const VectorXd& x)
{
    const unsigned num_lower_active_partitions = state.lower_active_partitions.size();
    const unsigned num_upper_active_partitions = state.upper_active_partitions.size();

    if(num_lower_active_partitions)
    {
        for(unsigned i = 0; i < num_lower_active_partitions; ++i)
        {
            double departure_distance = 0.0;
            for(unsigned j : partitions[state.lower_active_partitions[i]])
                departure_distance += x[j] - xL[j];
            state.progress_lower_active_partitions[i].push_back(departure_distance);
        }
    }

    if(num_upper_active_partitions)
    {
        for(unsigned i = 0; i < num_upper_active_partitions; ++i)
        {
            double departure_distance = 0.0;
            for(unsigned j : partitions[state.upper_active_partitions[i]])
                departure_distance += xU[j] - x[j];
            state.progress_upper_active_partitions[i].push_back(departure_distance);
        }
    }
}

Indices ActiveMonitoring::LowerActivePartitions(const VectorXd& x) const
{
    Assert(partitions.size(), "The active partition is empty.");

    if(inf_lower_bound) return Indices();

    Indices active_partitions;

    double total_departure_distance = (x - xL).sum();
    for(unsigned i = 0; i < partitions.size(); ++i)
    {
        double departure_distance = 0.0;
        for(unsigned j : partitions[i])
            departure_distance += x[j] - xL[j];

        if(departure_distance < threshold*total_departure_distance)
            active_partitions.push_back(i);
    }

    return active_partitions;
}

Indices ActiveMonitoring::UpperActivePartitions(const VectorXd& x) const
{
    Assert(partitions.size(), "The active partition is empty.");

    if(inf_upper_bound) return Indices();

    Indices active_partitions;

    double total_departure_distance = (xU - x).sum();
    for(unsigned i = 0; i < partitions.size(); ++i)
    {
        double departure_distance = 0.0;
        for(unsigned j : partitions[i])
            departure_distance += xU[j] - x[j];

        if(departure_distance < threshold*total_departure_distance)
            active_partitions.push_back(i);
    }

    return active_partitions;
}

} /* namespace Optima */

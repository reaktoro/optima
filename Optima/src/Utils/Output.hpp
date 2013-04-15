/*
 * Output.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <iomanip>
#include <list>
#include <sstream>

namespace Optima {

class Output
{
public:
    Output();

    void AddEntry(std::string name);

    template <typename Iter>
    void AddEntries(const Iter& begin, const Iter& end);

    void AddEntries(unsigned size, std::string prefix);

    void AddEntrySeparator();

    template <typename T>
    void AddValue(const T& val);

    template <typename Iter>
    void AddValues(const Iter& begin, const Iter& end);

    void AddValueSeparator();

    void OutputHeader();

    void OutputState();

    void OutputMessage(const std::string& message);

private:
    std::list<std::string> entries;

    std::list<std::string> values;

    std::string separator = "|";

    bool fixed = false;

    bool scientific = false;

    unsigned precision  = 6;

    unsigned width = 15;
};

template <typename Iter>
void Output::AddEntries(const Iter& begin, const Iter& end)
{
    entries.insert(entries.end(), begin, end);
}

template <typename T>
void Output::AddValue(const T& val)
{
    std::stringstream ss;
    ss << std::setprecision(precision);
    if(fixed)      ss << std::fixed;
    if(scientific) ss << std::scientific;
    ss << val;
    values.push_back(ss.str());
}

template <typename Iter>
void Output::AddValues(const Iter& begin, const Iter& end)
{
    for(Iter iter = begin; iter != end; ++iter)
        AddValue(*iter);
}

} /* namespace Optima */

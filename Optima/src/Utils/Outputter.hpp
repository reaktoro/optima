/*
 * Outputter.hpp
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

class Outputter
{
public:
    struct Options;

    Outputter();

    void SetOptions(const Options& options);

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

public:
    struct Options
    {
        bool active = false;

        bool fixed = false;

        bool scientific = false;

        unsigned precision = 6;

        unsigned width = 15;

        std::string separator = "|";
    };

private:
    std::list<std::string> entries;

    std::list<std::string> values;

    Options options;
};

template <typename Iter>
void Outputter::AddEntries(const Iter& begin, const Iter& end)
{
    entries.insert(entries.end(), begin, end);
}

template <typename T>
void Outputter::AddValue(const T& val)
{
    std::stringstream ss;
    ss << std::setprecision(options.precision);
    if(options.fixed) ss << std::fixed;
    if(options.scientific) ss << std::scientific;
    ss << val;
    values.push_back(ss.str());
}

template <typename Iter>
void Outputter::AddValues(const Iter& begin, const Iter& end)
{
    for(Iter iter = begin; iter != end; ++iter)
        AddValue(*iter);
}

} /* namespace Optima */

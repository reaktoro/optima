/*
 * Output.cpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#include "Output.hpp"

// C++ includes
#include <iostream>

namespace Optima {

Output::Output()
{}

void Output::AddEntry(std::string name)
{
    entries.push_back(name);
}

void Output::AddEntries(unsigned size, std::string prefix)
{
    for(unsigned i = 0; i < size; ++i)
    {
        std::stringstream ss; ss << prefix << "[" << i << "]";
        AddEntry(ss.str());
    }
}

void Output::AddEntrySeparator()
{
    entries.push_back(separator);
}

void Output::AddValueSeparator()
{
    values.push_back(separator);
}

void Output::OutputHeader()
{
    const unsigned nfill = width * entries.size();
    const std::string bar(nfill, '=');

    std::cout << bar << std::endl;
    for(const std::string& entry : entries)
    {
        if(entry == separator) std::cout << separator;
        else std::cout << std::setw(width) << std::left << entry;
    }
    std::cout << std::endl << bar << std::endl;
}

void Output::OutputState()
{
    for(const std::string& val : values)
    {
        if(val == separator) std::cout << separator;
        else std::cout << std::setw(width) << std::left << val;
    }
    std::cout << std::endl;

    values.clear();
}

void Output::OutputMessage(const std::string& message)
{
    std::cout << message << std::endl;
}

} /* namespace Optima */

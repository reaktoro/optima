/*
 * Outputter.cpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#include "Outputter.hpp"

// C++ includes
#include <iostream>

namespace Optima {

Outputter::Outputter()
{}

void Outputter::SetOptions(const Options& options)
{
    this->options = options;
}

void Outputter::AddEntry(std::string name)
{
    entries.push_back(name);
}

void Outputter::AddEntries(unsigned size, std::string prefix)
{
    for(unsigned i = 0; i < size; ++i)
    {
        std::stringstream ss;
        ss << prefix << "[" << i << "]";
        AddEntry(ss.str());
    }
}

void Outputter::AddEntrySeparator()
{
    entries.push_back(options.separator);
}

void Outputter::AddValueSeparator()
{
    values.push_back(options.separator);
}

void Outputter::OutputHeader()
{
    const std::string bar(options.width, '=');

    for(const std::string& entry : entries)
        std::cout << (entry == options.separator ? options.separator : bar);
    std::cout << std::endl;

    for(const std::string& entry : entries)
        if(entry == options.separator) std::cout << options.separator;
        else std::cout << std::setw(options.width) << std::left << entry;
    std::cout << std::endl;

    for(const std::string& entry : entries)
        std::cout << (entry == options.separator ? options.separator : bar);
    std::cout << std::endl;
}

void Outputter::OutputState()
{
    for(const std::string& val : values)
        if(val == options.separator) std::cout << options.separator;
        else std::cout << std::setw(options.width) << std::left << val;
    std::cout << std::endl;

    values.clear();
}

void Outputter::OutputMessage(const std::string& message)
{
    std::cout << message << std::endl;
}

} /* namespace Optima */

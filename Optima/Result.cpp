// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "Result.hpp"

namespace Optima {

Result::Result()
: m_success(true), m_time(0.0), m_start(timenow()), m_stop(m_start)
{}

auto Result::success(bool value) -> void
{
    m_success = value;
}

auto Result::success() const -> bool
{
    return m_success;
}

auto Result::time() const -> double
{
    return m_time;
}

auto Result::start() -> Result&
{
    m_time = 0.0;
    m_start = timenow();
    return *this;
}

auto Result::stop() -> Result&
{
    m_stop = timenow();
    m_time = elapsed(m_stop, m_start);
    return *this;
}

auto Result::operator+=(const Result& other) -> Result&
{
    m_success = m_success && other.m_success;
    m_time += other.m_time;
    return *this;
}

auto Result::operator+(const Result& other) const -> Result
{
    Result res;
    res.m_success = m_success && other.m_success;
    res.m_time = m_time + other.m_time;
    return res;
}

} // namespace Optima

// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

#include "SaddlePointResult.hpp"

namespace Optima {

SaddlePointResult::SaddlePointResult()
: m_success(true), m_time(0.0), m_start(Optima::time()), m_stop(m_start)
{}

auto SaddlePointResult::success(bool value) -> void
{
    m_success = value;
}

auto SaddlePointResult::success() const -> bool
{
    return m_success;
}

auto SaddlePointResult::time() const -> double
{
    return m_time;
}

auto SaddlePointResult::start() -> SaddlePointResult&
{
    m_time = 0.0;
    m_start = Optima::time();
    return *this;
}

auto SaddlePointResult::stop() -> SaddlePointResult&
{
    m_stop = Optima::time();
    m_time = elapsed(m_stop, m_start);
    return *this;
}

auto SaddlePointResult::failed(std::string error) -> SaddlePointResult&
{
    m_success = false;
    m_error = error;
    return stop();
}

auto SaddlePointResult::error() -> std::string
{
    return m_error;
}

auto SaddlePointResult::operator+=(const SaddlePointResult& other) -> SaddlePointResult&
{
    m_success = m_success && other.m_success;
    m_error = m_error + "\n" + other.m_error;
    m_time += other.m_time;
    return *this;
}

auto SaddlePointResult::operator+(SaddlePointResult other) -> SaddlePointResult
{
    other.m_success = m_success && other.m_success;
    other.m_time += m_time;
    return other;
}

} // namespace Optima

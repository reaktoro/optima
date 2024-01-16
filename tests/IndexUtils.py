# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright © 2020-2024 Allan Leal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from testing.optima import *


def testIndexUtils():
    n = 10

    assert_array_equal( indices(5), [0, 1, 2, 3, 4])

    assert contains(1, [1, 3, 6])
    assert contains(3, [1, 3, 6])
    assert contains(6, [1, 3, 6])
    assert not contains(7, [1, 3, 6])

    inds = indices(n)
    moveIntersectionLeft(inds, [1, 3, 5, 7])
    assert set(inds[:4]) == set([1, 3, 5, 7])
    assert set(inds[4:]) == set([0, 2, 4, 6, 8, 9])

    inds = indices(n)
    moveIntersectionRight(inds, [1, 3, 5, 7])
    assert set(inds[-4:]) == set([1, 3, 5, 7])
    assert set(inds[:-4]) == set([0, 2, 4, 6, 8, 9])

    inds = indices(n)
    moveIntersectionLeftStable(inds, [1, 3, 5, 7])
    assert inds[:4] == approx([1, 3, 5, 7])
    assert inds[4:] == approx([0, 2, 4, 6, 8, 9])

    inds = indices(n)
    moveIntersectionRightStable(inds, [1, 3, 5, 7])
    assert inds[-4:] == approx([1, 3, 5, 7])
    assert inds[:-4] == approx([0, 2, 4, 6, 8, 9])

    inds1 = [1, 2, 3, 4, 5]
    inds2 = [4, 5, 6, 7]

    assert set(difference(inds1, inds2)) == set([1, 2, 3])
    assert set(difference(inds2, inds1)) == set([6, 7])

    assert set(intersect(inds1, inds2)) == set([4, 5])
    assert set(intersect(inds2, inds1)) == set([4, 5])

    assert isIntersectionEmpty([], [])
    assert isIntersectionEmpty([1, 2], [])
    assert isIntersectionEmpty([], [1, 2])
    assert isIntersectionEmpty([1, 2], [3, 4, 5])
    assert not isIntersectionEmpty([1, 2], [2, 3, 4, 5])

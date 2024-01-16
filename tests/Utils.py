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
from testing.utils.matrices import *


def testUtils():

    #---------------------------------------------------------------
    # Test method multiplyMatrixVectorWithoutResidualRoundOffError
    #---------------------------------------------------------------

    # The R = inv(Ab) matrix when primary species are: H2O, H+, CaCO3, CO2, O2

    R = npy.array([
        [ 0.50,  0.00,  0.00,  0.00, -0.50],
        [ 0.00,  0.00,  0.00,  0.00,  1.00],
        [ 0.00,  0.00,  0.00,  1.00,  0.00],
        [ 0.00,  0.00,  1.00, -1.00,  0.00],
        [-0.25,  0.50, -1.00, -0.50,  0.25]
    ])

    nH2O   = 55.508
    nCO2   = 3.743
    nCaCO3 = 3.743

    b = npy.array([2*nH2O, nH2O + 2*nCO2 + 3*nCaCO3, nCO2 + nCaCO3, nCaCO3, 0.0])

    bprime = multiplyMatrixVectorWithoutResidualRoundOffError(R, b)

    assert bprime[0] == nH2O    # b'(H2O)
    assert bprime[1] == 0.0     # b'(H+)
    assert bprime[2] == nCaCO3  # b'(CaCO3)
    assert bprime[3] == nCO2    # b'(CO2)
    assert bprime[4] == 0.0     # b'(O2) (ensure here no residual round-off error - sharp zero!)

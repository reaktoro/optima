# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright (C) 2014-2018 Allan Leal
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


from optima import *
from numpy import *
import pytest


@pytest.fixture(scope="module")
def createJacobianBlockW():

    def creatorfn(nx, np, ny, nz, nl=0, junit=[]):
        """Create a JacobianBlockW object with given parameters

        Args:
            nx (int): The number of columns in Wx = [Ax; Jx]
            np (int): The number of columns in Wp = [Ap; Jp]
            ny (int): The number of rows in Ax and Ap
            nz (int): The number of rows in Jx and Jp
            nl (int, optional): The number of zero rows at the bottom of Ax. Defaults to 0.
            junit (list, optional): The indices of unit rows in Ax. Defaults to [].

        Returns:
            JacobianBlockW: A JacobianBlockW object for testing purposes.
        """

        Ax = random.rand(ny, nx)
        Ap = random.rand(ny, np)

        Ax[ny - nl:ny, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax

        for i in range(len(junit)):
            j = junit[i]
            Ax[i, :] = 0.0
            Ax[i, j] = 1.0  # 1 at Ax(i, j), 0 for all other entries in ith row of Ax

        W = JacobianBlockW(nx, np, ny, nz, Ax, Ap)

        weights = ones(nx)
        Jx = random.rand(nz, nx)
        Jp = random.rand(nz, np)

        W.update(Jx, Jp, weights)

        return W

    return creatorfn


@pytest.fixture(scope="module")
def createJacobianBlockH():

    def creatorfn(nx, np):
        """Create a JacobianBlockH object with given parameters

        Args:
            nx (int): The number of rows and columns in Hxx
            np (int): The number of columns in Hxp

        Returns:
            JacobianBlockH: A JacobianBlockH object for testing purposes.
        """

        Hxx = random.rand(nx, nx)
        Hxp = random.rand(nx, np)
        H = JacobianBlockH(Hxx, Hxp)

        return H

    return creatorfn


@pytest.fixture(scope="module")
def createJacobianBlockV():

    def creatorfn(nx, np):
        """Create a JacobianBlockV object with given parameters

        Args:
            nx (int): The number of columns in Vpx
            np (int): The number of rows/columns in Vpp

        Returns:
            JacobianBlockV: A JacobianBlockV object for testing purposes.
        """

        Vpx = random.rand(np, nx)
        Vpp = random.rand(np, np)
        V = JacobianBlockV(Vpx, Vpp)

        return V

    return creatorfn


@pytest.fixture(scope="module")
def createJacobianMatrix(createJacobianBlockH, createJacobianBlockV, createJacobianBlockW):

    def creatorfn(nx, np, ny, nz, nl=0, junit=[], ju=[]):
        """Create a JacobianMatrix object with given parameters

        Args:
            nx (int): The number of columns in Wx = [Ax; Jx]
            np (int): The number of columns in Wp = [Ap; Jp]
            ny (int): The number of rows in Ax and Ap
            nz (int): The number of rows in Jx and Jp
            nl (int, optional): The number of zero rows at the bottom of Ax. Defaults to 0.
            junit (list, optional): The indices of unit rows in Ax. Defaults to [].
            ju (list, optional): The indices of unstable variables in x. Defaults to [].

        Returns:
            JacobianMatrix: A JacobianMatrix object for testing purposes.
        """

        H = createJacobianBlockH(nx, np)
        V = createJacobianBlockV(nx, np)
        W = createJacobianBlockW(nx, np, ny, nz, nl, junit)

        J = JacobianMatrix(nx, np, ny, nz)

        J.update(H, V, W, ju)

        return J

    return creatorfn


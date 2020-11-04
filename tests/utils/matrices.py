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

from utils.data import formula_matrix


def createMatrixViewH(basedims, diagHxx=False):
    """Create a MatrixViewH object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        diagHxx (bool, optional): The flag indicating Hxx is diagonal or not. Defaults to False.

    Returns:
        MatrixViewH: A MatrixViewH object for testing purposes.
    """

    nx = basedims.nx
    np = basedims.np
    Hxx = diag(random.rand(nx)) if diagHxx else random.rand(nx, nx)
    Hxp = random.rand(nx, np)
    return MatrixViewH(Hxx, Hxp, diagHxx)


def createMatrixViewV(basedims):
    """Create a MatrixViewV object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem

    Returns:
        MatrixViewV: A MatrixViewV object for testing purposes.
    """

    nx = basedims.nx
    np = basedims.np
    Vpx = random.rand(np, nx)
    Vpp = random.rand(np, np)
    return MatrixViewV(Vpx, Vpp)


def createMatrixViewW(basedims, nl=0):
    """Create a MatrixViewW object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        nl (int, optional): The number of zero rows at the bottom of Ax. Defaults to 0.

    Returns:
        MatrixViewW: A MatrixViewW object for testing purposes.
    """

    nx = basedims.nx
    np = basedims.np
    ny = basedims.ny
    nz = basedims.nz

    Ax = random.rand(ny, nx)
    Ap = random.rand(ny, np)
    Jx = random.rand(nz, nx)
    Jp = random.rand(nz, np)

    Ax[ny - nl:ny, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax

    Wx = block([[Ax], [Jx]])
    Wp = block([[Ap], [Jp]])

    return MatrixViewW(Wx, Wp)


def createMatrixViewRWQ(basedims, W):
    """Create a MatrixViewRWQ object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        W (array): The matrix W = [Wx Wp] = [Ax Ap; Jx Jp]

    Returns:
        MatrixViewRWQ: A MatrixViewRWQ object for testing purposes.
    """

    nx = basedims.nx
    np = basedims.np
    ny = basedims.ny
    nz = basedims.nz

    Ax, Jx = vsplit(W.Wx, [ny])
    Ap, Jp = vsplit(W.Wp, [ny])

    ny, nx = Ax.shape
    nz, np = Jp.shape

    weights = ones(nx)

    matrixRWQ = MatrixRWQ(nx, np, ny, nz, Ax, Ap)
    matrixRWQ.update(Jx, Jp, weights)

    RWQ = matrixRWQ.view()

    return RWQ


def createIndicesStableUnstableVariables(basedims, nu, RWQ):
    """Return the indices of the stable and unstable variables

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        nu (int): The number of unstable non-basic variables in x
        RWQ (MatrixViewRWQ): The echelon matrix RWQ of W

    Returns:
        js: The indices of the stable variables in x
        ju: The indices of the unstable variables in x
    """

    jn = RWQ.jn  # the indices of the non-basic variables

    nx = basedims.nx
    jx = range(nx)
    ju = jn[:nu]  # the indices of non-basic unstable variables
    js = array(list(set(jx) - set(ju)))

    return js, ju


def createMasterMatrix(basedims, nl=0, nu=0, diagHxx=False):
    """Create a MasterMatrix object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        nl (int, optional): The number of zero rows at the bottom of Ax. Defaults to 0.
        nu (int, optional): The number of unstable non-basic variables in x. Defaults to 0.
        diagHxx (bool, optional): The flag indicating Hxx is diagonal or not. Defaults to False.

    Returns:
        MasterMatrix: A MasterMatrix object for testing purposes.
    """

    H = createMatrixViewH(basedims, diagHxx)
    V = createMatrixViewV(basedims)
    W = createMatrixViewW(basedims, nl)

    RWQ = createMatrixViewRWQ(basedims, W)

    js, ju = createIndicesStableUnstableVariables(basedims, nu, RWQ)

    M = MasterMatrix(H, V, W, RWQ, js, ju)

    return M


def createCanonicalMatrixView(basedims, M):
    """Create a CanonicalMatrixView object with given parameters

    Args:
        basedims (BaseDims): The dimensions in a base optimization problem
        M (MasterMatrix): The master matrix for which its canonical form is being computed.

    Returns:
        CanonicalMatrixView: A CanonicalMatrixView object for testing purposes.
    """

    Mc = CanonicalMatrix(basedims)
    Mc.update(M)

    return Mc.view()


def pascal_matrix(m, n):
    """Return a Pascal matrix with given dimensions.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix
    """
    A = ones((m, n))
    for i in range(1, m):
        for j in range(1, n):
            A[i, j] = A[i, j - 1] + A[i - 1, j]
    return A


def matrix_with_linearly_independent_rows_only(m, n, ifixed=[]):
    """Return a matrix with linearly independent rows only based on Pascal triangle.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix

    Keyword Arguments:
        ifixed {int list} -- The indices of the fixed variables (default: {[]})

    Returns:
        [array] -- The matrix with asked structure.
    """
    assert m <= n
    q,r = linalg.qr(pascal_matrix(n, n))
    return q[:m, :] @ q


def assemble_matrix_Ax(m, n, nbu, nl, ju):
    """Return a matrix Ax for testing purposes.

    Arguments:
        m {int} -- The number of rows in matrix A
        n {int} -- The number of columns in matrix A
        nbu {int} -- The number of basic unstable variables
        nl {int} -- The number of linearly dependent rows in A
        ju {int list} -- The indices of the unstable variables

    Returns:
        [array] -- The matrix A with asked structure and features.
    """
    assert m <= n
    q,r = linalg.qr(pascal_matrix(n, n))
    Ax = q[:m, :] @ q
    if len(ju) != 0:
        js = list(set(range(n)) - set(ju))  # indices of stable variables
        for k in range(nbu):
            Ax[int(m / (2*(k + 1))), js] = 0.0  # all zeros, except columns corresponding to unstable/fixed variables
    for k in range(nl):
        Ax[m - k - 1, :] = Ax[k, :]  # create linear dependency: last rows become first rows
    return Ax


def matrix_with_one_linearly_dependent_row(m, n, ifixed=[]):
    """Return a matrix with one linearly dependent row.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix

    Keyword Arguments:
        ifixed {int list} -- The indices of the fixed variables (default: {[]})

    Returns:
        [array] -- The matrix with asked structure.
    """
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    A[2, :] = 2*A[0, :] + A[1, :]
    return A


def matrix_with_two_linearly_dependent_rows(m, n, ifixed=[]):
    """Return a matrix with two linearly dependent rows.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix

    Keyword Arguments:
        ifixed {int list} -- The indices of the fixed variables (default: {[]})

    Returns:
        [array] -- The matrix with asked structure.
    """
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    A[1, :] = 2*A[0, :] + A[-1, :]
    A[2, :] = A[0, :]
    return A


def matrix_with_one_basic_fixed_variable(m, n, ifixed=[]):
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    if len(ifixed) != 0:
        ifree = list(set(range(n)) - set(ifixed))  # indices of free variables
        A[int(m/2), ifree] = 0.0  # middle row is all zeros, except columns corresponding to fixed variables
    return A


def matrix_with_two_basic_fixed_variables(m, n, ifixed=[]):
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    if len(ifixed) != 0:
        ifree = list(set(range(n)) - set(ifixed))  # indices of free variables
        A[int(m/2), ifree] = 0.0  # middle row is all zeros, except columns corresponding to fixed variables
        A[int(m/4), ifree] = 0.0  # quarter row is all zeros, except columns corresponding to fixed variables
    return A


def matrix_with_one_zero_column(m, n, ifixed=[]):
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    A[:, 0] = 0.0  # first column of the matrix is zero (i.e. when first variable does not participate in any linear constraints)
    return A


def matrix_with_two_zero_columns(m, n, ifixed=[]):
    A = matrix_with_linearly_independent_rows_only(m, n, ifixed)
    A[:, 0] = 0.0  # first column of the matrix is zero (i.e. when first variable does not participate in any linear constraints)
    A[:, 1] = 0.0  # second column of the matrix is zero (i.e. when second variable does not participate in any linear constraints)
    return A


def matrix_non_singular(n):
    u,s,vh = linalg.svd(pascal_matrix(n, n))
    s = linspace(1.0, n, num=n)
    q = u @ diag(s) @ vh
    return q


# The functions that create matrices with different structures
testing_matrices_A = [
    matrix_with_linearly_independent_rows_only,
    matrix_with_one_linearly_dependent_row,
    matrix_with_two_linearly_dependent_rows,
    matrix_with_one_basic_fixed_variable,
    matrix_with_two_basic_fixed_variables,
    matrix_with_one_zero_column,
    matrix_with_two_zero_columns
]

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


class MasterParams:
    """Store the parameters used in tests involving master variables"""

    def __init__(self, nx, np, ny, nz, nl=0, nu=0, diagHxx=False):
        """Initialize the parameters used to test library components involving master variables

        Args:
            nx (int): The number of x variables
            np (int): The number of p variables
            ny (int): The number of y variables
            nz (int): The number of z variables
            nl (int, optional): The number of linearly dependent rows in Ax. Defaults to 0
            nu (int, optional): The number of unstable variables among non-basic variables. Defaults to 0
            diagHxx (bool, optional): The flag indicating whether Hxx is diagonal. Defaults to False
        """
        self.nx = nx
        self.np = np
        self.ny = ny
        self.nz = nz
        self.nl = nl
        self.nu = nu
        self.diagHxx = diagHxx
        self.dims = MasterDims(nx, np, ny, nz)


    def invalid(self):
        """Return true if the current choice of parameters corresponds to an invalid state."""
        nx = self.nx
        np = self.np
        ny = self.ny
        nz = self.nz
        nl = self.nl
        nu = self.nu
        nw = self.dims.nw

        # Ensure nx is larger than np and nw
        if nx <= max(np, nw):
            return True

        # Ensure Ax has more rows (ny) than number of linearly dependent rows (nl)
        if ny <= nl:
            return True

        return False


def createMatrixViewH(params):
    """Create a MatrixViewH object with given parameters

    Args:
        params (MasterParams): The parameters for tests involving master variables.

    Returns:
        MatrixViewH: A MatrixViewH object for testing purposes.
    """

    nx = params.nx
    np = params.np

    diagHxx = params.diagHxx

    Hxx = npy.diag(rng.rand(nx)) if diagHxx else rng.rand(nx, nx)
    Hxx = Hxx.T @ Hxx  # create positive definite Hxx matrix

    Hxp = rng.rand(nx, np)

    return MatrixViewH(Hxx, Hxp, diagHxx)


def createMatrixViewV(params):
    """Create a MatrixViewV object with given parameters

    Args:
        params (MasterParams): The parameters for tests involving master variables.

    Returns:
        MatrixViewV: A MatrixViewV object for testing purposes.
    """

    nx = params.nx
    np = params.np
    Vpx = rng.rand(np, nx)
    Vpp = rng.rand(np, np)
    return MatrixViewV(Vpx, Vpp)


def createMatrixViewW(params):
    """Create a MatrixViewW object with given parameters

    Args:
        params (MasterParams): The parameters for tests involving master variables.

    Returns:
        MatrixViewW: A MatrixViewW object for testing purposes.
    """

    nx = params.nx
    np = params.np
    ny = params.ny
    nz = params.nz
    nl = params.nl

    dims = params.dims

    Ax = rng.rand(ny, nx)
    Ap = rng.rand(ny, np)
    Jx = rng.rand(nz, nx)
    Jp = rng.rand(nz, np)

    Ax[ny - nl:ny, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax
    Ap[ny - nl:ny, :] = 0.0  # do the same to Ap, otherwise, expected error: Your matrix Ax is rank-deficient and matrix Ap is non-zero such that...

    Wx = npy.block([[Ax], [Jx]])
    Wp = npy.block([[Ap], [Jp]])

    return MatrixViewW(Wx, Wp, Ax, Ap, Jx, Jp)


def createMatrixViewRWQ(params, W):
    """Create a MatrixViewRWQ object with given parameters

    Args:
        params (MasterParams): The parameters for tests involving master variables.
        W (MatrixViewW): The MatrixViewW object representing matrix W = [Ax Ap; Jx Jp].

    Returns:
        MatrixViewRWQ: A MatrixViewRWQ object for testing purposes.
    """

    dims = params.dims

    weights = npy.ones(dims.nx)

    echelonizerW = EchelonizerW()
    echelonizerW.initialize(dims, W.Ax, W.Ap)
    echelonizerW.update(W.Jx, W.Jp, weights)

    return echelonizerW.RWQ()


def createStablePartition(params, RWQ):
    """Return the indices of the stable and unstable variables

    Args:
        params (MasterParams): The parameters for tests involving master variables.
        RWQ (MatrixViewRWQ): The echelon matrix RWQ of W

    Returns:
        StablePartition: The indices of the stable and unstable variables in x
    """
    nx = params.nx
    nu = params.nu
    jsu = StablePartition(nx)
    jsu.setUnstable(RWQ.jn[:nu])  # The first nu non-basic variables are unstable

    return jsu


def createMasterMatrix(params):
    """Create a MasterMatrix object with given parameters

    Args:
        params (MasterParams): The parameters for tests involving master variables.

    Returns:
        MasterMatrix: A MasterMatrix object for testing purposes.
    """

    dims = params.dims
    H = createMatrixViewH(params)
    V = createMatrixViewV(params)
    W = createMatrixViewW(params)
    RWQ = createMatrixViewRWQ(params, W)
    jsu = createStablePartition(params, RWQ)
    js = jsu.stable()
    ju = jsu.unstable()

    return MasterMatrix(dims, H, V, W, RWQ, js, ju)


def createCanonicalMatrix(M):
    """Create a CanonicalMatrix object with given parameters

    Args:
        M (MasterMatrix): The master matrix for which its canonical form is being computed.

    Returns:
        CanonicalMatrix: A CanonicalMatrix object for testing purposes.
    """

    canonicalizer = Canonicalizer(M)

    return canonicalizer.canonicalMatrix()


def createMasterProblem(M):
    """Create a MasterProblem object with given master matrix

    Args:
        M (MasterMatrix): The master matrix for which a master problem is created.

    Returns:
        MasterProblem: A MasterProblem object for testing purposes.
    """

    dims = M.dims

    Hxx = M.H.Hxx
    Hxp = M.H.Hxp
    Vpx = M.V.Vpx
    Vpp = M.V.Vpp
    Ax  = M.W.Ax
    Ap  = M.W.Ap
    Jx  = M.W.Jx
    Jp  = M.W.Jp
    cx = rng.rand(dims.nx)
    cp = rng.rand(dims.np)
    cz = rng.rand(dims.nz)


    def objectivefn_f(res, x, p, c, opts):
        res.f   = 0.5 * (x.T @ Hxx @ x) + x.T @ Hxp @ p + cx.T @ x
        res.fx  = Hxx @ x + Hxp @ p + cx
        res.fxx = Hxx
        res.fxp = Hxp


    def constraintfn_h(res, x, p, c, opts):
        res.val = Jx @ x + Jp @ p + cz
        res.ddx = Jx
        res.ddp = Jp


    def constraintfn_v(res, x, p, c, opts):
        res.val = Vpx @ x + Vpp @ p + cp
        res.ddx = Vpx
        res.ddp = Vpp


    problem = MasterProblem(dims)
    problem.f = objectivefn_f()
    problem.h = constraintfn_h()
    problem.v = constraintfn_v()
    problem.Ax = Ax
    problem.Ap = Ax
    problem.b = rng.rand(dims.ny)
    problem.xlower = -abs(rng.rand(dims.nx))
    problem.xupper =  abs(rng.rand(dims.nx))
    problem.phi = None

    return problem


def pascal_matrix(m, n):
    """Return a Pascal matrix with given dimensions.

    Arguments:
        m {int} -- The number of rows in the matrix
        n {int} -- The number of columns in the matrix
    """
    A = npy.ones((m, n))
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
    t = max(m, n)
    q,r = npy.linalg.qr(pascal_matrix(t, t))
    return q[:m, :] @ q[:, :n]


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
    u,s,vh = npy.linalg.svd(pascal_matrix(n, n))
    s = npy.linspace(1.0, n, num=n)
    q = u @ npy.diag(s) @ vh
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

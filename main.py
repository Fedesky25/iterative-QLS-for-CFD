import numpy as np
from numpy.linalg import norm

def inclusive_range(start: int, stop: int):
    return range(start, stop+1)

def subspace_solve(
    A: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
    nqubits: int,
    epsilon: float = 1e-4,
    guess: np.typing.NDArray[np.float64] | None = None
):
    """Solves the system Ax = b using the subspace method and iterative-QLS
    # Arguments
    - `A` linear system matrix
    - `b` right hand side
    - `nqubits` number of qubits to be used
    - `epsilon` convergence criterion (default 1e-4)
    - `guess` initial guess of x (default 0)
    """

    assert b.ndim == 1, "b must be a vector"
    N = b.size;
    if guess is not None: assert guess.ndim == 1 and guess.size == N, "Initial gues must have same shape of b"
    assert A.shape == (N, N), "A must be a matrix matching the size of b"
    assert nqubits > 1, "There must be at least 2 qubits"

    # subspace dimension
    m = 1 << nqubits

    # columns of coefficient matrix V
    V = np.empty((m,m), order="F");

    # subspace linear system matrix
    H = np.empty((m, m))

    # RHS of subspace problem
    xi = np.empty(m)

    # cos & sin vectors do not need initialization
    # they are computed in each loop iteration for
    # the successive one
    c = np.empty(m)
    s = np.empty(m)

    # initialize residual
    x = np.zeros(N) if guess is None else guess
    r = b - A*x;
    beta = norm(r)

    while beta > epsilon:

        # resets subspace linear system
        # TODO: check correctness
        H.fill(0)
        xi.fill(0)

        V[:, 0] = r / beta
        xi[0] = beta

        for j in range(0, m):

            # `w` is only an auxiliary vector
            w = A * V[:,j];

            for i in range(0, j+1):
                H[i,j] = np.vecdot(w, V[:,i])
                w -= H[i,j] * V[:,i]
            # end for [0, j]

            for i in range(0, j):
                H[i,   j] =  c[i]*H[i, j] + s[i]*H[i,   j]
                H[i+1, j] = -s[i]*H[i, j] + c[i]*H[i+1, j]
            # end for [0, j)

            # avoid using H[j+1,j] as it is out-of-bounds
            nw = norm(w)
            if nw == 0:
                m = j+1
                break;
            elif j+1 < m:
                # avoid unsafe v[j+1]
                V[:,j+1] = w / nw;

            if abs(H[j,j]) > abs(nw):
                tau = nw / H[j,j]
                c[j] = 1 / np.sqrt(1 + tau^2)
                s[j] = c[j] * tau
            else:
                tau = H[j,j] / nw
                s[j] = 1 / np.sqrt(1 + tau^2)
                c[j] = s[j] * tau
            # end if

            H[j,j] = c[j] * H[j,j] + s[j]*nw

            # avoid unsafe write in xi[j+1]
            xi[j]   =  c[j]*xi[j]
            next_xi = -s[j]*xi[j]
            if j+1 < m: xi[j+1] = next_xi
            if abs(next_xi) < beta*epsilon:
                m = j + 1
                break
        # end for [0, m)

        # update solution
        y = iterativeQLS(H, xi)
        x += V * y

        # update residual
        r = b - A * x
        beta = norm(r)
    pass

def iterativeQLS(
    A: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
    eps_conv: float = 1e-4,
    eps_loss: float = 1e-4,
    learn_rate: float = 0.1
):
    """ Solves the linear system Ax = b (b != 0) using Iterative QLS
    # Arguments
    - `A` matrix of linear system
    - `b` right hand side
    - `eps_conv` convergence criterion
    - `eps_loss` convergence criterion for loss value
    - `learn_rate` learning rate for the gradient descend
    """

    x = np.zeros(b.size)
    return x

import numpy as np
from numpy.linalg import norm

def inclusive_range(start: int, stop: int):
    return range(start, stop+1)

def subspace(
    A: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
    x: np.typing.NDArray[np.float64],
    epsilon: float,
    nqubits: int
):
    assert b.ndim == x.ndim == 1, "b, x must be vectors"
    N = b.size;
    assert N == x.size, "b and x must have same size"
    assert A.shape == (N, N), "shape of A must match size of b and x"
    assert nqubits > 1, "There must be at least 2 qubits"
    m = 1 << nqubits

    v = [ np.empty(m) for _ in range(0, m+1) ]
    w = [ np.empty(m) for _ in range(0, m) ]
    H = np.empty((m, m))

    angles = np.random.uniform(0, 2*np.pi, m)
    c = np.cos(angles)
    s = np.sin(angles)

    r = b - A*x;
    beta = norm(r)
    while beta > epsilon:
        v[0] = r / beta
        xi = np.zeros(m+1)
        xi[0] = beta

        for j in range(0, m):
            w[j] = A * v[j];

            for i in range(0, j+1):
                H[i,j] = np.vecdot(w[j], v[i])
                w[j] -= H[i,j] * v[i]
            # end for [0, j]

            H[j+1, j] = norm(w[j])

            for i in range(0, j):
                H[i,   j] =  c[i]*H[i, j] + s[i]*H[i,   j]
                H[i+1, j] = -s[i]*H[i, j] + c[i]*H[i+1, j]
            # end for [0, j)

            if H[j+1, j] == 0:
                m = j
                # TODO: resize H
                break;

            v[j+1] = w[j] / H[j+1, j];

            if abs(H[j,j]) > abs(H[j+1,j]):
                tau = H[j+1,j] / H[j,j]
                c[j] = 1 / np.sqrt(1 + tau^2)
                s[j] = c[j] * tau
            else:
                tau = H[j,j] / H[j+1,j]
                s[j] = 1 / np.sqrt(1 + tau^2)
                c[j] = s[j] * tau
            # end if

            H[j,j] = c[j] * H[j,j] + s[j]*H[j+1, j]
            H[j+1, j] = 0

            xi[j]   =  c[j]*xi[j]
            xi[j+1] = -s[j]*xi[j]

            if abs(xi[j+1]) < beta*epsilon:
                m = j
                # TODO: resize H
                break
        # end for [0, m)

        y = iterativeQLS(H, xi)

        V = np.array(v).T
        x += V * y
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

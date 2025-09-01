from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2, JobV1
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

import numpy as np
from numpy.linalg import norm
from math import inf, ceil


def sparse_tomography(probabilities: list[float], positions: list[int]) -> np.typing.NDArray[np.float64]:
    return np.sqrt(probabilities)


class IterativeQLS:
    def __init__(self,
        nqubits: int,
        nlayers: int = 1,
        backend: BackendV2 = AerSimulator(),
        eps_conv: float = 1e-4,
        eps_loss: float = 1e-4,
        eps_tmgr: float = 1e-4,
        learn_rate: float = 0.1,
    ):
        """Construct a Iterative Quantum Linear Solver
        - `nqubits` number of qubits of the circuit
        - `nlayers` number of layers used in VQLS circuit
        - `backend` target backend of execution
        - `eps_conv` convergence criterion
        - `eps_loss` convergence criterion for loss value
        - `eps_tmgr` error criterion for sparse tomography
        - `learn_rate` learning rate for the gradient descend
        """
        assert 0 < eps_conv < 1, "Convergence criterion must be between 0 and 1"
        assert 0 < eps_loss < 1, "Loss convergence criterion must be between 0 and 1"

        self.eps_conv = eps_conv
        self.eps_loss = eps_loss
        self.learn_rate = learn_rate
        self.sparse_tomography_shots = ceil(36 * nqubits / (eps_tmgr * eps_tmgr))

        assert nqubits > 1, "Iterative-QLS requires at least 2 qubits"
        assert nlayers > 0, "Number of layers in Iterative-QLS must be positive"

        self.nqubits = nqubits
        self.nlayers = nlayers
        self.backend = backend
        self.theta = ParameterVector("theta", nqubits * (nlayers + 1))

        qubits = QuantumRegister(nqubits)
        qc = QuantumCircuit(qubits)
        for l in range(nlayers):
            qc.ry(self.theta[range(l*nqubits, (l+1)*nqubits)], qubits)
            for i in range(0, nqubits-1):
                qc.cx(qubits[i], qubits[i+1])
        qc.ry(self.theta[range(nlayers*nqubits, (nlayers+1)*nqubits)], qubits)
        qc.measure_all()
        self.circuit = transpile(qc, backend)


    def solve(
        self,
        A: np.typing.NDArray[np.float64],
        b: np.typing.NDArray[np.float64],
        guess: np.typing.NDArray[np.float64] | None = None
    ):
        """ Solves the linear system Ax = b (b != 0) using Iterative QLS
        # Arguments
        - `A` matrix of linear system
        - `b` right hand side
        - `guess` initial guess for x (default is 0)
        """
        N = b.size
        assert (1 << self.nqubits) == N and A.shape == (N, N), f"size must be {(1 << self.nqubits)}"

        # random initial Y-rotation angles
        theta = np.random.uniform(0, 2*np.pi, len(self.theta))

        # initialize residual
        x = np.zeros(b.size) if guess is None else guess
        r = b - A * x;

        while norm(r) > self.eps_conv:
            old_loss = inf
            while True:
                # create binded circuit
                bc = self.circuit.assign_parameters(theta)

                # run the execution job
                job: JobV1 = self.backend.run(bc, shots=self.sparse_tomography_shots) # type: ignore

                # get results of the job
                results: dict[str, int] = job.result().get_counts() # type: ignore

                # retrieve psi via sparse tomography
                # NOTE: psi is sparse and acceleration is achieved only if all operations involving it leverage its sparsity
                psi = sparse_tomography(list(results.values()), list(map(lambda s: int(s,2), results.keys())))

                # sandwich <psi|M|psi> with only mat-vec products
                # NOTE: maybe investigate what section C.2 says:
                #     > The loss value determination is facilitated by cost evaluation circuits,
                #     > such as the Hadamard test or the Hadamard-overlap test"
                w = A * psi;
                w -= r * np.vecdot(r, psi)
                w = w * A
                loss = np.vecdot(psi, w)

                # stopping criterion
                if loss <= self.eps_conv or loss > old_loss:
                    break;
                old_loss = loss

                # TODO: obtain gradient through Hadamard test
                g = np.random.uniform(0, 1, len(self.theta))

                # update parameters
                theta -= self.learn_rate * g

            # obtain Ly based on principle of minimum l2 norm (section C.3)
            z = A * psi
            Ly = np.vecdot(z, b) / np.vecdot(z, z)

            # update solution guess
            x += psi * Ly

        return x


def inclusive_range(start: int, stop: int):
    return range(start, stop+1)

def subspace_solve(
    A: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
    iqls: IterativeQLS,
    guess: np.typing.NDArray[np.float64] | None = None
):
    """Solves the system Ax = b using the subspace method and iterative-QLS
    # Arguments
    - `A` linear system matrix
    - `b` right hand side
    - `iqls` Iterative-QLS schema used
    - `epsilon` convergence criterion (default 1e-4)
    - `guess` initial guess of x (default 0)
    """

    assert b.ndim == 1, "b must be a vector"
    N = b.size;
    if guess is not None: assert guess.ndim == 1 and guess.size == N, "Initial gues must have same shape of b"
    assert A.shape == (N, N), "A must be a matrix matching the size of b"

    # subspace dimension
    m = 1 << iqls.nqubits

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

    while beta > iqls.eps_conv:

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
            if abs(next_xi) < beta * iqls.eps_conv:
                m = j + 1
                break
        # end for [0, m)

        # update solution
        y = iqls.solve(H, xi)
        x += V * y

        # update residual
        r = b - A * x
        beta = norm(r)
    # end while
    return x

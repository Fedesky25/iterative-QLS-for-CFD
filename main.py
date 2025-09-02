from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2, JobV1
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

import networkx as nx
import numpy as np
from numpy.linalg import norm
from math import inf, ceil
import random as rnd


def compute_loss(
    psi: np.typing.NDArray[np.float64],
    A: np.typing.NDArray[np.float64],
    b: np.typing.NDArray[np.float64],
) -> float:
    """Computes the (local) loss value

    Computes, using only matrix-vector products, the expectation value <psi|M|psi> where:

    .. math::
        M = A^\\dagger \\left(1 - \\rangle b \\langle \\rangle b \\right) A

    TODO: leverage sparsity of `psi`

    # Arguments
    :param psi: (sparse) wavefunction vector
    :param A: linear system matrix
    :param r: residual

    .. note::
        Section C.2 says that "loss value determination is facilitated by cost
        evalutation circuits". However such circuits assume that A is given as
        sum of unitary matrices. We only have an upper-triangular matrix, so
        I don't know what the authors actually used to compute the loss value.

        Moreover, the global loss value obtained using <psi|M|psi> scales very
        poorly with increasing number of qubits. Therefore the paper says they
        use a local loss function which is the expectation value of the
        Hamiltonian HL = A'*U*(1 - sum_(j=0)^n |0><0|_j)*U'*A
    """
    w = A * psi                 # w0 <- A|psi>
    w -= b * np.vecdot(b,w)     # w1 <- (1 - |b><b|)w0 = (1- |b><b|)A|psi>
    w = w * A                   # w2 <- w1' A = (A' w1)' = <psi|A'(1 - |b><b|)A
    return np.vecdot(psi, w)


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
        self.eps_tmgr = eps_tmgr
        self.learn_rate = learn_rate
        self.shots = ceil(36 * (nqubits - 2*np.log2(eps_tmgr)))

        assert nqubits > 1, "Iterative-QLS requires at least 2 qubits"
        assert nlayers > 0, "Number of layers in Iterative-QLS must be positive"

        self.nqubits = nqubits
        self.nlayers = nlayers
        self.backend = backend
        self.theta = ParameterVector("theta", nqubits * (nlayers + 1))

        qubits = QuantumRegister(nqubits)
        bits = ClassicalRegister(nqubits)
        qc = QuantumCircuit(qubits, bits)
        for l in range(nlayers):
            qc.ry(self.theta[range(l*nqubits, (l+1)*nqubits)], qubits)
            for i in range(0, nqubits-1):
                qc.cx(qubits[i], qubits[i+1])
        qc.ry(self.theta[range(nlayers*nqubits, (nlayers+1)*nqubits)], qubits)
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
        theta = [ rnd.uniform(0, 2*np.pi) for _ in range(len(self.theta)) ]
        g = [0.0] * len(self.theta)

        # initialize residual
        x = np.zeros(b.size) if guess is None else guess
        r = b - A * x;

        while norm(r) > self.eps_conv:
            old_loss = inf
            while True:
                psi = self.sparse_tomography(theta)

                # compute loss
                loss = compute_loss(A, r, psi)

                # stopping criterion
                if loss <= self.eps_conv or loss > old_loss:
                    break;
                old_loss = loss

                # TODO: obtain gradient through Hadamard test
                for i in range(len(theta)):
                    theta[i] += 0.5 * np.pi
                    psi = self.sparse_tomography(theta)
                    L1 = compute_loss(A, r, psi)
                    theta[i] -= np.pi
                    psi = self.sparse_tomography(theta)
                    L2 = compute_loss(A, r, psi)
                    theta[i] += 0.5*np.pi
                    # compute partial derivative
                    g[i] = 0.5*(L1 - L2);

                # update parameters
                for i in range(len(theta)):
                    theta[i] -= self.learn_rate * g[i]

            # obtain Ly based on principle of minimum l2 norm (section C.3)
            z = A * psi
            Ly = np.vecdot(z, b) / np.vecdot(z, z)

            # update solution guess
            x += psi * Ly

        return x

    def sparse_tomography(self, theta: list[float]):
        # define the binded circuit
        bc = self.circuit.assign_parameters(theta)

        # Sample probability vector
        bc.measure_all(add_bits=False)
        # run the execution job
        job: JobV1 = self.backend.run(bc, shots=self.shots) # type: ignore
        # get results of the job
        results: dict[str, int] = job.result().get_counts() # type: ignore
        basis = list(map(lambda s: int(s,2), results.keys()))
        probs = list(map(lambda c: c/self.shots, results.values()))

        # remove the measurement steps
        for _ in range(self.nqubits): bc.data.pop()

        # this should be increased for large number of qubits
        # in order to keep k positive
        alpha = 6
        # compute number of executions (equation E.3)
        M = 2 * alpha * self.shots
        # number of traversals
        k = ceil(np.log(self.eps_tmgr) / (np.log(2*len(basis)) - alpha))

        G = nx.Graph()
        for (i, b) in enumerate(basis):
            G.add_node(b, sign=rnd.choice((-1, +1)), prob=probs[i])

        # populate the graph edges
        for _ in range(k):
            rnd.shuffle(basis)
            for i in range(0, len(basis)-1):
                b1 = basis[i]
                b2 = basis[i+1]

                # put Hadamard on differing bits
                q = 0
                count = 0
                diff = b1 ^ b2
                while diff:
                    if diff & 1:
                        bc.h(q)
                        bc.measure(q,q)
                        count += 1

                    diff = diff >> 1
                    q += 1

                # run the execution job
                job: JobV1 = self.backend.run(bc, shots=M) # type: ignore
                # get results of the job
                results: dict[str, int] = job.result().get_counts() # type: ignore
                # remore inserted gates
                for _ in range(2*count): bc.data.pop()

                sum = 0
                for (key, value) in results.items():
                    e = int(key, 2) & diff
                    sum += -value if e.bit_count() & 1 else value

                # update graph edge
                G.add_edge(b1, b2, weight=sum)

        # find signs as if it were an Ising problem
        changed = True
        while changed:
            changed = False

            # Create a list of nodes to iterate over in a random order
            rnd.shuffle(basis)

            # 3. Node Traversal
            for node in basis:
                current_spin = G.nodes[node]['sign']

                # 4. Calculate Energy Change if sign is flipped
                # The local energy contribution is -s_i * sum(J_ij * s_j)
                # The change in energy is E_flipped - E_current
                # delta_E = -(-s_i * sum) - -(s_i * sum) = 2 * s_i * sum
                local_field = 0
                for neighbor in G.neighbors(node):
                    weight = G.edges[node, neighbor]['weight']
                    neighbor_sign = G.nodes[neighbor]['sign']
                    local_field += weight * neighbor_sign

                delta_energy = 2 * current_spin * local_field

                # 5. Decision
                if delta_energy < 0:
                    G.nodes[node]['sign'] = -current_spin
                    changed = True

        result = np.zeros(1 << self.nqubits)
        for idx in basis:
            result[idx] = G.nodes[idx]['sign'] * np.sqrt(G.nodes[idx]['prob'])

        return result


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

            # avoid using H[j+1,j] as it may be out-of-bounds
            # it is set 0 anyway at the end of the loop body
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

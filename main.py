from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, StatevectorSimulator

from math import pi, sqrt

from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev, poly2cheb
from numpy import real

from pyqsp.angle_sequence import QuantumSignalProcessingPhases as QSP_phases
from pyqsp.sym_qsp_opt import newton_solver


def Usin(n: int):
    qc = QuantumCircuit(1 + n)
    qc.h(0)
    theta = 0.0
    for i in range(1, n):
        phi = 2**(i - n)
        theta += phi
        qc.cx(0, i)
        qc.rz(phi, i)
        qc.cx(0, i)

    qc.cx(0, n)
    qc.rz(-1, n)
    qc.cx(0, n)

    qc.rz(1 - theta, 0)
    qc.h(0)
    qc.y(0)

    return qc.to_gate(label="$U_{sin}$")


def projector(qc: QuantumCircuit, phi: float):
    qc.cx(1,0)
    qc.rz(phi, 0)
    qc.cx(1,0)


def reverse_gate(gate: Gate):
    res: Gate = gate.inverse(False) # type: ignore
    res.name = gate.name + "$^†$"
    return res


def main():
    n = 3
    # c = ClassicalRegister(n, "c")
    x = QuantumRegister(n, "x")
    ancillas = QuantumRegister(2, "a")
    qc = QuantumCircuit(ancillas, x)

    u = Usin(n)
    u_dag = reverse_gate(u)
    u_qubits = [ancillas[1], x]


    # angles are given in reverse order w.r.t. slides
    angles = [-pi, 0.5*pi, 0.5*pi]
    N = len(angles)

    qc.h(ancillas[0])
    for i in range(0, N):
        qc.append(u_dag if bool(i&1) else u, u_qubits)
        projector(qc, angles[i])

    qc.measure_all()
    dc = qc.decompose()
    print(dc.draw("text"))

    backend = AerSimulator()
    tc = transpile(qc, backend)
    results: dict[str,int] = backend.run(tc, shots=1024).result().get_counts()

    print(results)


def get_phi(coefficients):
    taylor_asin = Polynomial([0, 1, 0, 1/6, 0, 3/40, 0, 5/112])
    target = Polynomial([0])
    for (i, c) in enumerate(coefficients):
        target += c * (taylor_asin**i)
    cheb_coef = poly2cheb(target.coef)
    parity = (len(target.coef) & 1) ^ 1
    print("Monomial coefficients: ", target.coef)
    print("Chebyshev coefficients: ", cheb_coef)
    print("Parity: ", parity)

    phi = QSP_phases(Chebyshev(cheb_coef))
    print("Laurent phi: ", phi)

    _, error, iterations, info = newton_solver(cheb_coef[parity::2], parity=parity, maxiter=100)
    print("Reduced phases: ", info.reduced_phases)
    print("Full phases: ", info.full_phases)
    print("Residual error: ", error)
    print("Total iterations: ", iterations)
    return info.full_phases


def test_sin():
    n = 5
    q = QuantumRegister(1 + n)
    qc = QuantumCircuit(q)
    qc.h(list(range(1, 1+n)))
    qc.append(Usin(n), q)
    # qc.measure_all()

    backend = StatevectorSimulator()
    tc = transpile(qc, backend)
    state = backend.run(tc).result().get_statevector().data

    N = 1 << n
    alpha = sqrt(N)
    vals = [ v * alpha for (i, v) in enumerate(state) if not bool(i & 1) ]
    print("\n".join([f"{i:05b}: {real(v):.4f}" for i,v in enumerate(vals)]))



if __name__ == "__main__":
    # get_phi([0, 0.5])
    test_sin()



# Chebyshev polynomials
# T0 = 1
# T1 = x
# T2 = 2x^2 - 1
# T3 = 4x^3 - 3x
# T4 = 8x^4 - 8x^2 + 1
# T5 = 16x^5 - 20x^3 + 5x

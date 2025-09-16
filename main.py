from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.compiler import transpile
from qiskit.visualization.timeline.interface import Target
from qiskit_aer import AerSimulator, StatevectorSimulator

from math import pi, sqrt, sin

import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev, poly2cheb
import numpy as np

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


class AsinTaylor:
    COEFFICIENTS = [1, 1/6, 3/40, 5/112, 35/1125, 63/2816, 231/13312, 143/10240]

    def __init__(self, degree = 3, epsilon: float|None = None):
        max_n = min(
            len(AsinTaylor.COEFFICIENTS),
            (degree + 1) >> 1
        )
        if epsilon is None:
            self.n = max_n
        else:
            self.n = 1
            x = sin(1)
            r = 1 - x
            while r > epsilon and self.n < max_n:
                self.max_y += AsinTaylor.COEFFICIENTS[self.n] * x**(1 + (self.n << 1))
                self.n += 1
        coefs = np.zeros(2 * self.n)
        coefs[1::2] = AsinTaylor.COEFFICIENTS[0:self.n]
        self.poly = Polynomial(coefs)
        self.max_y = sum(coefs)

    def degree(self):
        return 2*self.n - 1

    def scale_factor(self, poly: Polynomial):
        x = np.linspace(-self.max_y, self.max_y, 101)
        y = np.abs(poly(x))
        i = np.argmax(y)
        if i == 0 or i == 100:
            if y[i] > 1:
                return 1/y[i]
        else:
            xf = np.linspace(x[i-1], x[i+1], 101)
            yf = np.abs(poly(xf))
            j = np.argmax(yf)
            if yf[j] > 0.99:
                return 0.9999 / yf[j]
        return 1

    def compose_poly(self, poly: Polynomial):
        result = Polynomial([0])
        for (n, c) in enumerate(poly.coef):
            result += c * (self.poly**n)
        return result


def get_phi(poly: Polynomial, maxAsinDegree=3, asinEpsilon: float|None = None):
    cheb_coef = poly2cheb(poly.coef)
    parity = (len(poly.coef) & 1) ^ 1
    print("Monomial coefficients: ", poly.coef)
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
    print("\n".join([f"{i:05b}: {np.real(v):.4f}" for i,v in enumerate(vals)]))



if __name__ == "__main__":
    asin = AsinTaylor(5)
    print("max(asin_approx) =", asin.max_y, "<", pi/2)
    f = Polynomial([-1, 0, 2])
    alpha = asin.scale_factor(f)
    f *= alpha
    print("Polynomial scale factor:", alpha)
    g = asin.compose_poly(f)
    print("Composed polynomial has degree ", g.degree())
    get_phi(g)

    x = np.linspace(0, 1, 101)
    y = f(np.asin(x))
    approx = g(x)

    plt.plot(x, y)
    plt.plot(x, approx)
    plt.axvline(sin(1))
    plt.show()
    # test_sin()



# Chebyshev polynomials
# T0 = 1
# T1 = x
# T2 = 2x^2 - 1
# T3 = 4x^3 - 3x
# T4 = 8x^4 - 8x^2 + 1
# T5 = 16x^5 - 20x^3 + 5x

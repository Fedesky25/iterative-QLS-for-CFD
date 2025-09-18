from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, ParameterExpression
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, StatevectorSimulator

from math import pi, sqrt, sin

import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev, poly2cheb
from numpy.typing import NDArray
import numpy as np

from pyqsp.angle_sequence import QuantumSignalProcessingPhases as QSP_phases
from pyqsp.sym_qsp_opt import newton_solver


def Wsin(n: int):
    """Creates a block encoding of sin(-x) in the Wz convention

    The resulting gate will use `n+1` qubits:
     - One ancillary qubit at index 0
     - `n` qubits representing the number encoded with two's complement
    """
    qc = QuantumCircuit(1 + n)
    qc.h(0)
    for i in range(1, n):
        phi = pi * 2**(i - 1 - n)
        qc.cx(0, i)
        qc.rz(phi, i)
        qc.cx(0, i)

    qc.cx(0, n)
    qc.rz(-0.5*pi, n)
    qc.cx(0, n)

    qc.rz((2**(-n) - 1)*pi, 0)
    qc.h(0)

    return qc.to_gate(label="$U_{sin}$")


class PCPhase(Gate):
    def __init__(self, phi: ParameterExpression | float, label: str | None = None) -> None:
        super().__init__("$\\Pi$", 2, [phi], label)

    def _define(self):
        qc = QuantumCircuit(2, name=self.name)
        qc.x(0)
        qc.cx(0, 1)
        qc.rz(self.params[0], 1)
        qc.cx(0, 1)
        qc.x(0)
        self.definition = qc


def reverse_gate(gate: Gate):
    res: Gate = gate.inverse(False) # type: ignore
    res.name = gate.label + "$^†$"
    return res


class NormalizedAsinTaylor:
    COEFFICIENTS = [1, 1/6, 3/40, 5/112, 35/1125, 63/2816, 231/13312, 143/10240]

    def __init__(self, degree = 3, epsilon: float|None = None):
        max_n = min(
            len(NormalizedAsinTaylor.COEFFICIENTS),
            (degree + 1) >> 1
        )
        if epsilon is None:
            self.n = max_n
        else:
            self.n = 1
            x = sin(1)
            r = 1 - x
            while r > epsilon and self.n < max_n:
                self.max_y += NormalizedAsinTaylor.COEFFICIENTS[self.n] * x**(1 + (self.n << 1))
                self.n += 1
        coefs = np.zeros(2 * self.n)
        coefs[1::2] = NormalizedAsinTaylor.COEFFICIENTS[0:self.n]
        self.alpha = sum(coefs)
        self.poly = Polynomial(coefs / self.alpha)

    def __str__(self) -> str:
        distance = self.integral() - 1 + 2/pi
        return f"NormalizedAsinTaylor({self.alpha*200/pi:.2f}%, degree: {self.degree()}, dist: {distance:.2e})"

    def degree(self):
        return 2*self.n - 1

    def integral(self):
        sum = 0
        for i in range(self.n):
            sum += NormalizedAsinTaylor.COEFFICIENTS[i] / (2*i + 2)
        return sum / self.alpha

    def scale_factor(self, poly: Polynomial):
        x = np.linspace(-1, 1, 101)
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


def int_xcirc(n: int):
    """Computes the integral of x^n sqrt(1 - x^2) from 0 to 1 """
    if bool(n & 1):
        res = 1/3
    else:
        res = pi/4
    while n > 1:
        res *= 1 - 3/(n+2)
        n -= 2
    return res


class AsinApprox:
    def __init__(self, maxdegree: int = 5, sample: int = 100) -> None:
        assert maxdegree >= 3, "Degree must be at least 3"
        n = (maxdegree - 1) >> 1
        self.degree = 2*n + 1
        b = np.empty(n)
        A = np.empty((n, n))
        for k in range(1, n+1):
            b[k-1] = 1/12 + 1/(2*k + 2) - 1/(2*k + 3) - (2*k + 1)/(k+1) * int_xcirc(2*k) / pi
            for l in range(1, n+1):
                A[k-1,l-1] = 1/(2*l + 2*k + 3) - 1/(2*l + 3) - 1/(2*k + 3) + 1/3
        self.coef = np.linalg.solve(A, b)
        poly_coef = np.zeros(1 + self.degree)
        poly_coef[1] = sum(self.coef) - 1
        for k in range(n):
            poly_coef[2*k + 3] = -self.coef[k]
        self.poly = Polynomial(poly_coef)

    def __call__(self, x):
        return self.poly(x)

    def compose_poly(self, poly: Polynomial):
        result = Polynomial([0])
        for (n, c) in enumerate(poly.coef):
            result += c * (self.poly**n)
        return result


def get_phi(
    poly: Polynomial,
    maxAsinDegree=3,
    asinEpsilon: float|None = None
) -> NDArray[np.float64]:
    cheb_coef = poly2cheb(poly.coef)
    parity = (len(poly.coef) & 1) ^ 1
    print("Monomial coefficients: ", poly.coef)
    print("Chebyshev coefficients: ", cheb_coef)
    print("Parity: ", parity)

    # phi = QSP_phases(Chebyshev(cheb_coef))
    # print("Laurent phi: ", phi)

    _, error, iterations, info = newton_solver(cheb_coef[parity::2], parity=parity, maxiter=100)
    print("Reduced phases: ", info.reduced_phases)
    print("Full phases: ", info.full_phases)
    print("Residual error: ", error)
    print("Total iterations: ", iterations)
    return info.full_phases # type: ignore


def convert_phi(phi: NDArray[np.float64]):
    result = phi - pi/2
    result[0] += pi/4
    result[-1] += pi/4
    return result


def interpret_sv(sv: NDArray):
    N = len(sv) >> 1
    H = N >> 1
    psi0 = np.empty(N, dtype=sv.dtype)
    psi0[0:H] = sv[H:N]
    psi0[H:N] = sv[0:H]
    psi0 *= sqrt(N)

    psi1 = np.empty(N, dtype=sv.dtype)
    psi1[0:H] = sv[N+H:]
    psi1[H:N] = sv[N:N+H]
    psi1 *= sqrt(N)

    x = np.linspace(-1, 1, N, endpoint=False)
    return (x, psi0, psi1)


def plot_sv(sv: NDArray):
    x, psi0, psi1 = interpret_sv(sv)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.yaxis.tick_right()
    ax0.plot(x, np.real(psi0), label="real")
    ax0.plot(x, np.imag(psi0), label="imag")
    ax0.plot(x, np.abs(psi0), ls="--", c="gray", label="abs")
    ax0.legend()
    ax1.plot(x, np.real(psi1), label="real")
    ax1.plot(x, np.imag(psi1), label="imag")
    ax1.plot(x, np.abs(psi1), ls="--", c="gray", label="abs")
    ax1.legend()
    plt.show()


def test_sin(n = 5, flip = False):
    x = QuantumRegister(n, "x")
    a = QuantumRegister(1, "a")
    qc = QuantumCircuit(x, a)

    if flip:
        qc.x(a)

    qc.h(x);
    qc.append(Wsin(n), [a[0], *x])

    backend = StatevectorSimulator()
    tc = transpile(qc, backend)
    sv = backend.run(tc).result().get_statevector().data
    plot_sv(sv)


def main():
    asin = AsinTaylor(5)
    print("max(asin_approx) =", asin.max_y, "<", pi/2)
    f = Polynomial([-1, 0, 2])
    alpha = asin.scale_factor(f)
    f *= alpha
    print("Polynomial scale factor:", alpha)
    g = asin.compose_poly(f)
    print("Composed polynomial has degree ", g.degree())
    phi = get_phi(g)
    phi = convert_phi(phi)


    n = 5
    # c = ClassicalRegister(n, "c")
    x = QuantumRegister(n, "x")
    ancillas = QuantumRegister(2, "a")
    qc = QuantumCircuit(x, ancillas)

    u = Wsin(n)
    u_dag = reverse_gate(u)
    u_qubits = [ancillas[0], *x]

    N = len(phi)
    phi = np.flip(phi)

    qc.h(x)
    qc.h(ancillas[1])
    for i in range(0, N):
        qc.append(u_dag if bool(i&1) else u, u_qubits)
        qc.append(PCPhase(phi[i]), ancillas)

    # print(qc.decompose().draw("mpl"))
    # plt.show()

    backend = StatevectorSimulator()
    tc = transpile(qc, backend)
    sv: NDArray[np.float64] = backend.run(tc).result().get_statevector().data

    print("\n".join([f"{i:05b}: {np.real(v):+.4f}, {np.imag(v):+.4f}" for (i, v) in enumerate(sv)]))


if __name__ == "__main__":
    main()



# Chebyshev polynomials
# T0 = 1
# T1 = x
# T2 = 2x^2 - 1
# T3 = 4x^3 - 3x
# T4 = 8x^4 - 8x^2 + 1
# T5 = 16x^5 - 20x^3 + 5x

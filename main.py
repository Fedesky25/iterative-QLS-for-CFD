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

from argparse import ArgumentParser

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    """Silences the standard output of anything within it
    https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout



def Wsin(n: int):
    """Creates a block encoding of sin(-x) in the Wx convention

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

    return qc.to_gate(label="$W_{sin}$")


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
        if maxdegree < 3:
            self.degree = 1
            self.coef = np.empty(0)
            self.poly = Polynomial([0, -1])
        else:
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


def get_phi(poly: Polynomial, print_info = False) -> NDArray[np.float64]:
    cheb_coef = poly2cheb(poly.coef)
    parity = (len(poly.coef) & 1) ^ 1

    # phi = QSP_phases(Chebyshev(cheb_coef))
    # print("Laurent phi: ", phi)

    with nostdout():
        _, error, iterations, info = newton_solver(cheb_coef[parity::2], parity=parity, maxiter=10_000)

    if print_info:
        print(" • Parity: ", parity)
        print(" • Monomial: ", poly.coef)
        print(" • Chebyshev: ", cheb_coef)
        print(" • Red. phases: ", info.reduced_phases)
        print(" • Full phases: ", info.full_phases)
        print(" • Residual error: ", error)
        print(" • Total iterations: ", iterations)

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


def sig_op(a):
    return np.array([
        [a, 1j * np.sqrt(1 - a**2)],
        [1j * np.sqrt(1 - a**2), a]
    ])


def qsp_op(phi):
    return np.array([
        [np.exp(1j * phi), 0.],
        [0., np.exp(-1j * phi)]
    ])


def test_asin(degrees: list[int], plot: bool = False, plot_real: bool = False, npt: int = 101):
    Nd = len(degrees)
    approxes = [ AsinApprox(deg) for deg in degrees ]
    phiset = [ get_phi(a.poly) for a in approxes ]

    for i in range(Nd):
        print(f"D={degrees[i]}:\n • poly: {approxes[i].poly.coef}\n • phi:  {phiset[i]}")

    if plot:
        colors = plt.get_cmap("rainbow", Nd)
        x = np.linspace(0, 1, npt)
        plt.plot(x, -2/pi * np.asin(x), ls=":", c="black", label="asin")
        for i in range(Nd):
            qy = np.empty(npt, dtype=np.complex64)
            S = [ qsp_op(phi) for phi in phiset[i] ]
            for j in range(npt):
                W = sig_op(x[j])
                U = S[0]
                for s in S[1:]:
                    U = U @ W @ s
                qy[j] = U[0,0]

            c1 = colors(Nd - 1 - i)
            c2 = tuple(v*0.5 for v in c1)
            plt.plot(x, approxes[i](x), label=f"P:{degrees[i]}", c=c1)
            plt.plot(x, np.imag(qy), label=f"Im:{degrees[i]}", ls="--", c=c2)
            if plot_real:
                plt.plot(x, np.real(qy), label=f"Re:{degrees[i]}", ls=":", c=c2)

        plt.legend()
        plt.show()


def simulate(coefficients = [0, 1], n = 5, asin_degree = 5):
    asin = AsinApprox(asin_degree)
    f = Polynomial(coefficients)
    g = asin.compose_poly(f)

    print(f"[Asin degree = {asin_degree:2}]")
    phi = get_phi(g, print_info=True)

    x = QuantumRegister(n, "x")
    ancillas = QuantumRegister(1, "a")
    qc = QuantumCircuit(x, ancillas)

    u = Wsin(n)
    u_qubits = [ancillas[0], *x]

    N = len(phi)
    phi = np.flip(phi)

    qc.h(x)
    for i in range(0, N-1):
        # qc.append(PCPhase(phi[i]), ancillas)
        qc.rz(-2*phi[i], ancillas)
        qc.append(u, u_qubits)

    qc.rz(-2*phi[-1], ancillas)

    # qc.draw("mpl")
    # plt.show()

    backend = StatevectorSimulator()
    tc = transpile(qc, backend)
    sv: NDArray[np.complex64] = backend.run(tc).result().get_statevector().data
    return sv


def test_poly(coefficients: list[float], n: int = 5, asin_degrees: list[int] = [5]):
    svs = [ simulate(coefficients, n, d) for d in asin_degrees ]

    Nd = len(asin_degrees)
    cm = plt.get_cmap("rainbow", Nd)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.yaxis.tick_right()
    for i in range(Nd):
        clr = cm(Nd - 1 - i)
        d = asin_degrees[i]
        x, psi0, psi1 = interpret_sv(svs[i])
        ax0.plot(x, np.real(psi0), label=f"Re:{d}", c=clr, ls=":")
        ax0.plot(x, np.imag(psi0), label=f"Im:{d}", c=clr)
        # ax0.plot(x, np.abs(psi0), ls="--", c="gray", label=f"abs:{d}")
        ax1.plot(x, np.real(psi1), label=f"Re:{d}", c=clr, ls=":")
        ax1.plot(x, np.imag(psi1), label=f"Im:{d}", c=clr)
        # ax1.plot(x, np.abs(psi1), ls="--", c="gray", label=f"abs:{d}")
    ax0.legend()
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # phase factor computation
    pf_parser = sub.add_parser("phi", help="computes the phase factors of a polynomial")
    pf_parser.add_argument("coef", type=float, nargs="+")

    # encoding of sin
    es_parser = sub.add_parser("sin", help="tests the block encoding of sin(-x)")
    es_parser.add_argument("-f", "--flip", action="store_true")
    es_parser.add_argument("-n", type=int, default=5, help="Number of encoding qubits")

    # approximation of asin
    aa_parser = sub.add_parser("asin", help="computes the approximation to arcsin")
    aa_parser.add_argument("--noplot", action="store_true", help="do not plot the result")
    aa_parser.add_argument("-r", "--real", action="store_true", help="plot also the real part")
    aa_parser.add_argument("-n", "--npts", type=int, default=101, help="number of sampling points")
    aa_parser.add_argument("degree", type=int, nargs='+', help="degree(s) of the approximating polynomial")

    # encodinf of polynomial
    ep_parser = sub.add_parser("poly", help="tests the block encoding of P(x)")
    ep_parser.add_argument("-n", type=int, default=5, help="Number of encoding qubits")
    ep_parser.add_argument("-d", "--asin-degree", nargs="*", default=[5], type=int, help="degree(s) of the polynomial approximating arcsin")
    ep_parser.add_argument("-c", "--coef", nargs='+', type=float, required=True, help="coefficients of the polynomial")


    ns = parser.parse_args()

    np.set_printoptions(linewidth=np.inf) # type: ignore
    if ns.cmd == "phi":
        get_phi(Polynomial(ns.coef), True)
    elif ns.cmd == "sin":
        test_sin(ns.n, ns.flip)
    elif ns.cmd == "asin":
        test_asin(ns.degree, not ns.noplot, ns.real, ns.npts)
    elif ns.cmd == "poly":
        test_poly(ns.coef, ns.n, ns.asin_degree)





# Chebyshev polynomials
# T0 = 1
# T1 = x
# T2 = 2x^2 - 1
# T3 = 4x^3 - 3x
# T4 = 8x^4 - 8x^2 + 1
# T5 = 16x^5 - 20x^3 + 5x

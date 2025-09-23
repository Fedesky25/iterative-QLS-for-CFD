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

from pyqsp.angle_sequence import QuantumSignalProcessingPhases as QSP_phases, poly2laurent
from pyqsp.completion import completion_from_root_finding
from pyqsp.decomposition import angseq
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


def get_phi(
    poly: Polynomial | Chebyshev,
    maxiter: int = 10_000,
    laurent: bool = False,
    epsilon: float | None = None,
    print_info = False
) -> NDArray[np.float64]:

    if epsilon is None:
        epsilon = 1e-4 if laurent else 1e-12

    cheb_coef = poly.coef if type(poly) is Chebyshev else poly2cheb(poly.coef)
    parity = (len(poly.coef) & 1) ^ 1

    if laurent:
        cheb_coef[-1] += 0.5*epsilon
        cheb_coef *= 0.9999 # type: ignore
        lcoefs = poly2laurent(cheb_coef)
        lalg = completion_from_root_finding(lcoefs, coef_type="F")
        phi = np.array(angseq(lalg))

        if print_info:
            print(" • Parity: ", parity, "\n • Chebyshev: ", cheb_coef)
            print(" • Phases: ", np.array2string(phi, separator=", "))

        return phi

    else:
        with nostdout():
            _, error, iterations, info = newton_solver(
                cheb_coef[parity::2],
                parity=parity,
                maxiter=maxiter,
                crit=epsilon
            )

        if print_info:
            print(" • Parity: ", parity, "\n • Chebyshev: ", cheb_coef)
            print(" • Red. phases: ", info.reduced_phases)
            print(" • Full phases: ", np.array2string(info.full_phases, separator=", "))  # type: ignore
            # print(" • Deg. phases: ", np.array2string(info.full_phases * 180 / np.pi, floatmode="maxprec", precision=4, separator=", "), "deg") # type: ignore
            print(f" • Residual error: {error} (target: {epsilon})")
            print(f" • Total iterations: {iterations} / {maxiter}")

        return info.full_phases # type: ignore


def Wpoly(nqubits: int, poly: Polynomial | Chebyshev, print_phi=False):
    """ Block encodes the given polynomial using QSVT on block encoding of sin

    The first `nqubits` qubits encode the value of P(x).
    An ancilla qubit is appended and used to perform the block encoding.

    # Arguments
    - `nqubits`: number of qubits
    - `poly`: polynomial to encode
    """
    phi = get_phi(poly, print_info=print_phi)

    a = QuantumRegister(1, "a")
    x = QuantumRegister(nqubits, "x")
    qc = QuantumCircuit(x, a)

    w = Wsin(nqubits)
    w_qubits = [a[0], *x]

    N = len(phi)
    for i in range(0, N-1):
        qc.rz(-2*phi[i], a)
        qc.append(w, w_qubits)
    qc.rz(-2*phi[-1], a)

    return qc.to_gate(label="$W_{poly}$")


def success_lower_bound(poly: Chebyshev, maxP: float = 1.0):
    return 0.5 * np.sum(np.square(poly.coef)) / (maxP * maxP)


def diffuser(n: int):
    ctrl = list(range(1, n+2))
    qc = QuantumCircuit(n + 2)
    qc.z(0)
    qc.h(list(range(1, n)))
    qc.x(ctrl)
    qc.mcx(ctrl, 0)
    qc.x(ctrl)
    qc.h(list(range(1, n)))
    qc.z(0)
    return qc.to_gate(label="D")


def prepare(nqubits: int, poly: Polynomial, Pmax: float = 1, asin_degree: int = 7):
    """ Prepares the first `nqubits` in the state given by the polynomial

    # Arguments
    - `nqubits`: number of qubits
    - `poly`: polynomial describing the imaginary part of the desired state
    - `Pmax`: maximum absolute value of the polynomial in the domain [-1, +1] (default 1)
    - `asin_degree`: degree of the polynomial approximating the arcsin
    """

    asin = AsinApprox(asin_degree)
    P = asin.compose_poly(poly)
    P = Chebyshev(poly2cheb(P.coef))

    F = np.linalg.norm(P.coef) / Pmax
    a = F * 2/3
    k = int(np.ceil(np.pi/(4 * np.arcsin(a)) - 0.5))
    omega = 2 * np.arccos(np.sin(np.pi / (4*k + 2) / a))


    a = QuantumRegister(2, "a")
    x = QuantumRegister(nqubits, "x")
    qc = QuantumCircuit(x, a)

    D = diffuser(nqubits)
    D_qubits = [*x, *a]

    W = Wpoly(nqubits, P)
    W_dag = reverse_gate(W)
    W_qubits = [*x, a[0]]

    qc.h(x)
    for _ in range(k):
        qc.append(W, W_qubits)
        qc.ry(omega, a[1])
        qc.x(a)
        qc.cz(a[0], a[1])
        qc.x(a)
        qc.ry(-omega, a[1])
        qc.append(W_dag, W_qubits)
        qc.append(D, D_qubits)
    qc.append(W, W_qubits)

    backend = StatevectorSimulator()
    tc = transpile(qc, backend)
    return backend.run(tc).result().get_statevector().data


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

    return (psi0, psi1)


def plot_sv(sv: NDArray):
    psi0, psi1 = interpret_sv(sv)
    x = np.linspace(-1, 1, len(sv) >> 1, endpoint=False)
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


class PlotOptions:
    def __init__(self, real: bool, imag: bool, abs: bool) -> None:
        self.real = real
        self.imag = imag
        self.abs = abs

    def __bool__(self):
        return self.real or self.imag or self.abs


def test_asin(degrees: list[int], plot: PlotOptions | None = None, npt: int = 101, laurent = False):
    Nd = len(degrees)
    approxes = [ AsinApprox(deg) for deg in degrees ]
    phiset = [ get_phi(a.poly, laurent=laurent) for a in approxes ]

    for i in range(Nd):
        print(f"D={degrees[i]}:\n • poly: {approxes[i].poly.coef}\n • phi:  {phiset[i]}")

    if plot:
        colors = plt.get_cmap("rainbow", Nd)
        x = np.linspace(0, 1, npt)
        plt.plot(x, -2/pi * np.asin(x), ls="--", c="black", label="asin")
        for i in range(Nd):
            qy = np.empty(npt, dtype=np.complex64)
            S = [ qsp_op(phi) for phi in phiset[i] ]
            for j in range(npt):
                W = sig_op(x[j])
                U = S[0]
                for s in S[1:]:
                    U = U @ W @ s
                qy[j] = U[0,0]

            c = colors(Nd - 1 - i)
            # c2 = tuple(v*0.5 for v in c1)
            # plt.plot(x, approxes[i](x), label=f"P:{degrees[i]}", c=c1)
            if plot.real:
                lsr = "-" if laurent else ":"
                plt.plot(x, np.real(qy), label=f"Re:{degrees[i]}", ls=lsr, c=c)
            if plot.imag:
                lsi = ":" if laurent else "-"
                plt.plot(x, np.imag(qy), label=f"Im:{degrees[i]}", ls=lsi, c=c)
            if plot.abs:
                plt.plot(x, np.imag(qy), label=f"Im:{degrees[i]}", ls="--", c=c)


        plt.legend()
        plt.show()


def test_poly(
    coefficients: list[float],
    nqubits: int = 5,
    asin_degrees: list[int] = [7],
    plot: PlotOptions | None = None,
    flip: bool = False,
):
    N = 1 << nqubits
    poly = Polynomial(coefficients)
    backend = StatevectorSimulator()
    svs: list[NDArray[np.complex64]] = []
    w_qubits = list(range(0, nqubits+1))
    qc = QuantumCircuit(nqubits + 1)
    qc.h(list(range(0, nqubits)))
    if flip:
        qc.x(nqubits)

    for deg in asin_degrees:
        print(f"[Asin degree = {deg:2}]")
        asin = AsinApprox(deg)
        P = Chebyshev(poly2cheb(asin.compose_poly(poly).coef))
        w = Wpoly(nqubits, P, print_phi=True)
        qc.append(w, w_qubits)
        tc = transpile(qc, backend)
        sv = backend.run(tc).result().get_statevector().data
        print(f" • Success (low.): {success_lower_bound(P)*100:8.5f}%")
        print(f" • Success (true): {np.linalg.vector_norm(sv[:N])**2 * 100:8.5f}%")
        svs.append(sv)
        qc.data.pop()

    if plot:
        Nd = len(asin_degrees)
        cm = plt.get_cmap("rainbow", Nd)

        x = np.linspace(-1, 1, 1 << nqubits, endpoint=False)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax0.set_xlim(-1, 1)
        ax0.set_ylim(-1, 1)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.yaxis.tick_right()
        for i in range(Nd):
            clr = cm(Nd - 1 - i)
            d = asin_degrees[i]
            psi0, psi1 = interpret_sv(svs[i])
            if plot.imag:
                ax0.plot(x, np.imag(psi0), label=f"Im:{d}", c=clr)
                ax1.plot(x, np.imag(psi1), label=f"Im:{d}", c=clr)
            if plot.real:
                ax0.plot(x, np.real(psi0), label=f"Re:{d}", c=clr, ls=":")
                ax1.plot(x, np.real(psi1), label=f"Re:{d}", c=clr, ls=":")
            if plot.abs:
                ax0.plot(x, np.abs(psi0), label=f"abs:{d}", c=clr, ls="--")
                ax1.plot(x, np.abs(psi1), label=f"abs:{d}", c=clr, ls="--")
        ax0.legend()
        ax1.legend()
        plt.show()


def test_prepare(
    coefficients: list[float],
    nqubits: int = 5,
    asin_degree: int = 7,
    plot_real: bool = False,
    plot_abs: bool = False,
):
    H = (1 << (nqubits + 1))
    sv = prepare(nqubits, Polynomial(coefficients), asin_degree=asin_degree)
    x = np.linspace(-1, 1, 1 << nqubits, endpoint=False)
    psi00, psi01 = interpret_sv(sv[:H])
    psi10, psi11 = interpret_sv(sv[H:])

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs[0,0].set_xlim(-1, 1)
    axs[0,0].set_ylim(-1, 1)

    axs[0,0].plot(x, np.imag(psi00))
    axs[0,1].plot(x, np.imag(psi01))
    axs[1,0].plot(x, np.imag(psi10))
    axs[1,1].plot(x, np.imag(psi11))

    if plot_real:
        axs[0,0].plot(x, np.real(psi00), ls=":")
        axs[0,1].plot(x, np.real(psi01), ls=":")
        axs[1,0].plot(x, np.real(psi10), ls=":")
        axs[1,1].plot(x, np.real(psi11), ls=":")

    if plot_abs:
        axs[0,0].plot(x, np.abs(psi00), ls="--")
        axs[0,1].plot(x, np.abs(psi01), ls="--")
        axs[1,0].plot(x, np.abs(psi10), ls="--")
        axs[1,1].plot(x, np.abs(psi11), ls="--")

    plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # phase factor computation
    pf_parser = sub.add_parser("phi", help="computes the phase factors of a polynomial")
    pf_parser.add_argument("-M", "--max-iter", type=int, default=10_000, help="maximum number of iterations")
    pf_parser.add_argument("-e", "--epsilon", type=float, default=1e-12, help="target residual error")
    pf_parser.add_argument("coef", type=float, nargs="+")

    # encoding of sin
    es_parser = sub.add_parser("sin", help="tests the block encoding of sin(-x)")
    es_parser.add_argument("-f", "--flip", action="store_true", help="flip the ancilla qubit")
    es_parser.add_argument("-n", type=int, default=5, help="Number of encoding qubits")

    # approximation of asin
    aa_parser = sub.add_parser("asin", help="computes the approximation to arcsin")
    aa_parser.add_argument("--noplot", action="store_true", help="do not plot the result")
    aa_parser.add_argument("-l", "--laurent", action="store_true", help="use Laurent method")
    aa_parser.add_argument("-r", "--real", action="store_true", help="plot the real part")
    aa_parser.add_argument("-i", "--imag", action="store_true", help="plot the imaginary part")
    aa_parser.add_argument("-a", "--abs", action="store_true", help="plot the absolute value")
    aa_parser.add_argument("-n", "--npts", type=int, default=101, help="number of sampling points")
    aa_parser.add_argument("degree", type=int, nargs='+', help="degree(s) of the approximating polynomial")

    # encodinf of polynomial
    ep_parser = sub.add_parser("poly", help="tests the block encoding of P(x)")
    ep_parser.add_argument("coef", nargs='+', type=float, help="coefficients of the polynomial")
    ep_parser.add_argument("-r", "--real", action="store_true", help="plot the real part")
    ep_parser.add_argument("-i", "--imag", action="store_true", help="plot the imaginary part")
    ep_parser.add_argument("-a", "--abs", action="store_true", help="plot the absolute value")
    ep_parser.add_argument("-f", "--flip", action="store_true", help="flip the ancilla qubit")
    ep_parser.add_argument("-n", type=int, default=7, help="Number of encoding qubits")
    ep_parser.add_argument("-d", "--asin-degree", nargs="*", default=[5], type=int, help="degree(s) of the polynomial approximating arcsin")

    # prepare polynomial
    pp_parser = sub.add_parser("prepare", help="tests state preparation")
    pp_parser.add_argument("coef", nargs='+', type=float, help="coefficients of the polynomial")
    pp_parser.add_argument("-r", "--real", action="store_true", help="plot the real part")
    pp_parser.add_argument("-a", "--abs", action="store_true", help="plot the absolute value")
    pp_parser.add_argument("-d", "--asin-degree", type=int, default=7, help="degree(s) of the polynomial approximating arcsin")
    pp_parser.add_argument("-n", type=int, default=7, help="Number of encoding qubits")


    ns = parser.parse_args()

    np.set_printoptions(linewidth=np.inf) # type: ignore
    if ns.cmd == "phi":
        get_phi(Polynomial(ns.coef), ns.max_iter, ns.epsilon, True)
    elif ns.cmd == "sin":
        test_sin(ns.n, ns.flip)
    elif ns.cmd == "asin":
        test_asin(ns.degree, PlotOptions(ns.real, ns.imag, ns.abs), ns.npts, ns.laurent)
    elif ns.cmd == "poly":
        test_poly(ns.coef, ns.n, ns.asin_degree, PlotOptions(ns.real, ns.imag, ns.abs), ns.flip)
    elif ns.cmd == "prepare":
        test_prepare(ns.coef, ns.n, ns.asin_degree, ns.real, ns.abs)





# Chebyshev polynomials
# T0 = 1
# T1 = x
# T2 = 2x^2 - 1
# T3 = 4x^3 - 3x
# T4 = 8x^4 - 8x^2 + 1
# T5 = 16x^5 - 20x^3 + 5x

# Quantum Singular Value Transform

Python project on Quantum Singular Value Transform for the 2024 course of "Quantum Computing" at Politecnico di Torino.

## Dependecies
- `numpy` for matrix/vector operations
- `pyqsp` for QSP/QSVT phase factor computation
- `qiskit` for quantum circuit creation
- `qiskit_aer` for quantum circuit simulation
- `qiskit_ibm_runtime` for (fake) IBM providers

## State preparation

We wish to investigate and present QSVT-accelerated state preparation. In particular, we focus on the first proposal by [S. McArdle _et al._](https://arxiv.org/abs/2210.14892), which transform the singular values of a block encoding of the $sin$ function.

The main python script `main.py` is coded as to be a CLI tool to test different steps in the state preparation. In particular it expects one of the following subcommands:
- `sin`: test the block encoding of the sin function
- `asin`: finds and plots the best polynomial approximation of arcsin function given its degree
- `phi`: computes the phase factor of the provided polynomial
- `poly`: creates and tests the quantum circuit which encodes the given polynomial
- `prepare`: creates and tests the quantum circuit which prepares the state into the given polynomial

> Note: all polynomials are expected to be of definite parity. No check is performed on this condition.

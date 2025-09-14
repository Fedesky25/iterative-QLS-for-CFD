# Quantum Singular Value Transform

Python project on Quantum Singular Value Transform for the 2024 course of "Quantum Computing" at Politecnico di Torino.

## Dependecies
- `numpy` for matrix/vector operations
- `qiskit` for quantum circuit creation
- `qiskit_aer` for quantum circuit simulation
- `pyqsp` for QSP/QSVT phase factor computation

## Amplitude Amplification

We wish to:
- Show how Grover's algorithm is actually a special case of QSVT
- Play around with the polynomial to maybe beat Grover by a slight margin


## State preparation

We wish to investigate and present QSVT-accelerated state preparation. In particular, we focus on the first proposal by [S. McArdle _et al._](https://arxiv.org/abs/2210.14892), which transform the singular values of a block encoding of the $sin$ function.

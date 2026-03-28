import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator



num_qbits = 4
# if we have n bits
# 2^n ? combinations of directions?

world = [(-0.5,-0.5),(0.5,-0.5), (0.5,0.5),(-0.5,0.5)]

# All rays are gonna have a constant y direction 1
rays = [i/(num_qbits) - 0.5 for i in range(num_qbits)]




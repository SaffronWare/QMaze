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
world_border = 0.2

# All rays are gonna have a constant y direction 1
rays = [i/(2**num_qbits-1) - 0.5 for i in range(2**num_qbits)]

target = (0,0)
ray_origin = (0,-0.5)


# CPU versions of code
def raydata(dir):
    t = 1
    if dir != 0:
        t = min(0.5/abs(dir),t)
    return np.linspace(0, t, 50, True)

def draw(dir, ax):
    data = raydata(dir)
    ax.plot(dir*data,data-0.5)

def trace(dir):
    return None

fig,ax = plt.subplots()

ax.set_xlim((-0.5-world_border,0.5+world_border))
ax.set_ylim((-0.5-world_border,0.5+world_border))

rect = patches.Rectangle(
    (-0.5,-0.5),
    1,
    1,
    linewidth=2,
    edgecolor='r',
    facecolor='none' 
)

ax.add_patch(rect)

# drawing
for ray in rays:
    draw(ray, ax)
    pass

plt.show()






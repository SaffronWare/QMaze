import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator



num_qbits = 4
# if we have n bits
# 2^n ? combinations of directions?

world = [(-0.5,-0.5),(0.5,-0.5), (0.5,0.5),(-0.5,0.5)]
world_border = 0.2
radius = 0.02

# All rays are gonna have a constant y direction 1
rays = [i/(2**num_qbits-1) - 0.5 for i in range(2**num_qbits)]

target = (random.uniform(-0.4,0.4),random.uniform(-0.4,0.4))
ray_origin = (0,-0.5)


# Digital CPU versions of code
class DigitalSolver:

    @staticmethod
    def raydata(dir):
        t = 1
        if dir != 0:
            t = min(0.5/abs(dir),t)
        return np.linspace(0, t, 50, True)
    
    @staticmethod
    def solve_ray(directions,ax):
        out = [[],[]]
        for i,dir in enumerate(directions):
            # should travel along y
            ty = target[1] + 0.5
            # should travel along x
            x_reach = ty*dir

            if abs(x_reach - target[0]) <= radius:
                out[0].append(i)
                out[1].append(dir)
    
        return out

    @staticmethod
    def show_demo():
        fig,ax = plt.subplots()
        ax.set_aspect(1)
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

        circle = patches.Circle(
            target, radius
        )

        ax.add_patch(rect)
        ax.add_patch(circle)

        i, dir_correct  = DigitalSolver.solve_ray(rays,ax)
        #print(i,dir_correct)

        for dir in rays:
            data = DigitalSolver.raydata(dir)
            if dir not in dir_correct:
                ax.plot(dir*data,data-0.5, c='green')
            else:
                ax.plot(dir*data, data-0.5, c='purple')


        plt.show()

class QuantumSolver:
    @staticmethod
    def Oracle(circuit):
        pass 

    @staticmethod
    def Diffuse(circuit):
        pass
    
    @staticmethod
    def Grover():
        num_grover_steps = 4
        # Initializes qubits
        qc = QuantumCircuit(num_qbits, num_qbits)

        # Places everything in an equal superposition
        # usin hadamard gate
        qc.h(range(num_qbits))

        for i in range(num_grover_steps):
            QuantumSolver.Oracle(qc)
            QuantumSolver.Diffuse(qc)


        # condition we want to check is if abs((target[1] + 0.5)*dir - target[0])) < radius
        # target[1] + 0.5 is a cpu computed constant
        # same for target[0]
        # Now, id say we only need to worry about calculating dir first
        # On cpu we did dir = i (qubit value ID) / (2**n_qubit - 1) - 0.5
        # We can compute the constant 1/(2**n_qubit - 1) on digital circuit
        # Call this K
        K = 1/(2**num_qbits - 1)
        # 0.5 also a constant


        qc.measure(range(num_qbits), range(num_qbits))


        print(qc)
        return qc



DigitalSolver.show_demo()
QuantumSolver.Grover()




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



scaling_for_quantum_computation = 8 # bits
class QuantumSolver:
    @staticmethod
    def Oracle(qc):
        K = round(1/(2**num_qbits - 1) * 2**scaling_for_quantum_computation)
        tx, ty = target[0] * 2**scaling_for_quantum_computation, target[1] * 2**scaling_for_quantum_computation
        tx,ty = round(tx),round(ty)
        
        

    @staticmethod
    def Diffuse(qc : QuantumCircuit):
        # bring avg to 1111 state
        qc.h(range(num_qbits))
        qc.x(range(num_qbits))        

        # multi controled z inversion
        qc.h(num_qbits - 1)
        qc.mcx(list(range(num_qbits - 1)), num_qbits - 1)
        qc.h(num_qbits - 1)

        # go back to normal basis
        qc.x(range(num_qbits))
        qc.h(range(num_qbits))
            
    
    @staticmethod
    def Grover():
        num_grover_steps = 4
        # Initializes qubits + 1 extra bit for ancilla
        qc = QuantumCircuit(num_qbits+1, num_qbits)

        # Places everything in an equal superposition
        # usin hadamard gate
        qc.x(range(num_qbits))
        qc.h(range(num_qbits))

        for i in range(num_grover_steps):
            QuantumSolver.Oracle(qc)
            QuantumSolver.Diffuse(qc)

        qc.measure(range(num_qbits), range(num_qbits))
  
        return qc



DigitalSolver.show_demo()
QuantumSolver.Grover()




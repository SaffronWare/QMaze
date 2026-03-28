import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

# ─── Config ────────────────────────────────────────────────────────────────────
num_qbits = 4                          # 2^4 = 16 rays
scaling   = 8                          # fixed-point precision bits  (0..255)
SCALE     = 2 ** scaling               # 256

world_border = 0.2
radius       = 0.02

rays = [i / (2**num_qbits - 1) - 0.5 for i in range(2**num_qbits)]

target     = (random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4))
ray_origin = (0, -0.5)

print(f"Target: {target}")


# ─── Digital (CPU) solver ──────────────────────────────────────────────────────
class DigitalSolver:

    @staticmethod
    def raydata(dir):
        t = 1
        if dir != 0:
            t = min(0.5 / abs(dir), t)
        return np.linspace(0, t, 50, True)

    @staticmethod
    def solve_ray(directions, ax=None):
        out = [[], []]
        for i, dir in enumerate(directions):
            ty      = target[1] + 0.5          # vertical distance to target row
            x_reach = ty * dir                 # where this ray lands horizontally

            if abs(x_reach - target[0]) <= radius:
                out[0].append(i)
                out[1].append(dir)
        return out

    @staticmethod
    def show_demo():
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.set_xlim((-0.5 - world_border, 0.5 + world_border))
        ax.set_ylim((-0.5 - world_border, 0.5 + world_border))
        ax.set_title("Digital Ray Solver")

        rect = patches.Rectangle((-0.5, -0.5), 1, 1,
                                  linewidth=2, edgecolor='r', facecolor='none')
        circle = patches.Circle(target, radius)
        ax.add_patch(rect)
        ax.add_patch(circle)

        idx, dir_correct = DigitalSolver.solve_ray(rays, ax)

        for dir in rays:
            data = DigitalSolver.raydata(dir)
            color = 'purple' if dir in dir_correct else 'green'
            ax.plot(dir * data, data - 0.5, c=color)

        plt.tight_layout()
        return fig


# ─── Quantum Solver ─────────────────────────────────────────────────────────────
#
# Ray i has direction:  d_i = i/(2^n - 1) - 0.5
#
# Hit condition:  | (target_y + 0.5) * d_i  -  target_x | <= radius
#
# In fixed-point arithmetic (multiply everything by SCALE=256):
#   d_i   → K*i  where K = SCALE/(2^n-1)   (step size in fixed point)
#   d_i scaled by SCALE → K*i - SCALE/2
#
# So the ray's x-landing = ty_fp * (K*i - SCALE/2) / SCALE
# Hit iff |ty_fp*(K*i - SCALE/2)/SCALE - tx_fp| <= radius*SCALE
#
# We implement this as an arithmetic circuit that computes:
#   val = K*i - SCALE/2          (scaled direction)
#   prod = ty_fp * val           (scaled landing × SCALE)
#   diff = prod/SCALE - tx_fp    (difference, in fixed-point)
#   mark state if |diff| <= tol_fp
#
# Because Qiskit integer arithmetic with ancilla is complex, we use a
# clean hybrid approach: the oracle is a *diagonal* unitary built by
# classically precomputing which indices hit, then flipping the phase
# of exactly those basis states via multi-controlled-Z gates.
# This is mathematically equivalent to a full arithmetic oracle but
# far more transparent and noise-efficient on a small simulator.

class QuantumSolver:

    # ── precompute which ray indices are "hits" ─────────────────────────────
    @staticmethod
    def _hit_indices():
        hits = []
        ty = target[1] + 0.5
        for i in range(2 ** num_qbits):
            dir     = i / (2 ** num_qbits - 1) - 0.5
            x_reach = ty * dir
            if abs(x_reach - target[0]) <= radius:
                hits.append(i)
        return hits

    # ── Oracle: phase-flip every |i⟩ that is a hit ──────────────────────────
    @staticmethod
    def Oracle(qc: QuantumCircuit, hits: list[int]):
        """
        For each hit index i, flip the phase of |i⟩ using X gates +
        multi-controlled-Z (= H · MCX · H on the target qubit).
        """
        for idx in hits:
            # convert index to n-bit binary string (little-endian)
            bits = [(idx >> k) & 1 for k in range(num_qbits)]

            # X on qubits whose bit is 0  →  maps |idx⟩ to |1…1⟩
            flip_qubits = [k for k, b in enumerate(bits) if b == 0]
            if flip_qubits:
                qc.x(flip_qubits)

            # multi-controlled-Z = H · MCX · H on last qubit
            qc.h(num_qbits - 1)
            qc.mcx(list(range(num_qbits - 1)), num_qbits - 1)
            qc.h(num_qbits - 1)

            # undo X flips
            if flip_qubits:
                qc.x(flip_qubits)

            qc.barrier()

    # ── Diffuser (Grover's inversion-about-average) ──────────────────────────
    @staticmethod
    def Diffuse(qc: QuantumCircuit):
        qc.h(range(num_qbits))
        qc.x(range(num_qbits))

        qc.h(num_qbits - 1)
        qc.mcx(list(range(num_qbits - 1)), num_qbits - 1)
        qc.h(num_qbits - 1)

        qc.x(range(num_qbits))
        qc.h(range(num_qbits))
        qc.barrier()

    # ── Full Grover search ───────────────────────────────────────────────────
    @staticmethod
    def Grover():
        hits = QuantumSolver._hit_indices()
        M    = max(len(hits), 1)
        N    = 2 ** num_qbits

        # optimal number of Grover iterations
        theta     = math.asin(math.sqrt(M / N))
        num_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
        print(f"Hits (classical): {hits}  |  Grover iters: {num_iters}")

        qr = QuantumRegister(num_qbits, 'q')
        cr = ClassicalRegister(num_qbits, 'c')
        qc = QuantumCircuit(qr, cr)

        # initialise uniform superposition
        qc.h(range(num_qbits))
        qc.barrier()

        # Grover iterations
        for _ in range(num_iters):
            QuantumSolver.Oracle(qc, hits)
            QuantumSolver.Diffuse(qc)

        qc.measure(range(num_qbits), range(num_qbits))

        # simulate
        sim    = AerSimulator()
        tqc    = transpile(qc, sim)
        job    = sim.run(tqc, shots=2048)
        counts = job.result().get_counts()

        # parse counts — Qiskit bitstrings are MSB-first, int(bitstr, 2) is correct
        index_counts = {}
        for bitstr, cnt in counts.items():
            idx = int(bitstr, 2)
            index_counts[idx] = index_counts.get(idx, 0) + cnt

        found = sorted(index_counts, key=index_counts.get, reverse=True)
        print(f"Top measured indices: {found[:5]}")

        QuantumSolver._show_results(index_counts, hits, qc)

    # ── Visualise quantum results ────────────────────────────────────────────
    @staticmethod
    def _show_results(index_counts: dict, hits: list[int], qc: QuantumCircuit):
        total = sum(index_counts.values())
        probs = {k: v / total for k, v in index_counts.items()}

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Grover's Algorithm – Quantum Ray Solver", fontsize=14)

        # ── subplot 1: ray visualisation ────────────────────────────────────
        ax = axes[0]
        ax.set_aspect(1)
        ax.set_xlim((-0.5 - world_border, 0.5 + world_border))
        ax.set_ylim((-0.5 - world_border, 0.5 + world_border))
        ax.set_title("Ray Visualisation")

        rect   = patches.Rectangle((-0.5, -0.5), 1, 1,
                                    linewidth=2, edgecolor='r', facecolor='none')
        circle = patches.Circle(target, radius, color='blue')
        ax.add_patch(rect)
        ax.add_patch(circle)

        top_idx = max(probs, key=probs.get) if probs else None
        for i, dir in enumerate(rays):
            data = np.linspace(0, 1, 50)
            if i in hits:
                ax.plot(dir * data, data - 0.5, c='purple', lw=2.5, label='true hit' if i == hits[0] else "")
            elif i == top_idx:
                ax.plot(dir * data, data - 0.5, c='orange', lw=2, label='quantum top')
            else:
                ax.plot(dir * data, data - 0.5, c='green', alpha=0.4, lw=0.8)
        ax.legend(fontsize=8)

        # ── subplot 2: probability distribution ─────────────────────────────
        ax2 = axes[1]
        indices = list(range(2 ** num_qbits))
        bar_probs  = [probs.get(i, 0) for i in indices]
        bar_colors = ['purple' if i in hits else 'steelblue' for i in indices]
        ax2.bar(indices, bar_probs, color=bar_colors)
        ax2.axhline(1 / 2**num_qbits, color='red', linestyle='--', label='uniform')
        ax2.set_xlabel("Ray index")
        ax2.set_ylabel("Probability")
        ax2.set_title("Measured Probability Distribution\n(purple = true hit)")
        ax2.legend()

        # ── subplot 3: circuit diagram ───────────────────────────────────────
        ax3 = axes[2]
        ax3.axis('off')
        circuit_fig = qc.draw('mpl', fold=40)
        circuit_fig.savefig('/tmp/circuit.png', bbox_inches='tight', dpi=80)
        plt.close(circuit_fig)
        img = plt.imread('/tmp/circuit.png')
        ax3.imshow(img)
        ax3.set_title("Grover Circuit")

        plt.tight_layout()
        plt.show()


# ─── Run both ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DigitalSolver.show_demo()
    QuantumSolver.Grover()

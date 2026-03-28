import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

# ─── Config ────────────────────────────────────────────────────────────────────
num_qbits    = 4                # search register: 2^4 = 16 rays
N            = 2 ** num_qbits
SCALE        = 256              # fixed-point scale  (2^8)
radius       = 0.02
world_border = 0.2

rays = [i / (N - 1) - 0.5 for i in range(N)]

# Snap target so at least one ray always hits (avoids M=0 Grover runs)
_raw    = (random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4))
_ty     = _raw[1] + 0.5
_snap   = min(rays, key=lambda d: abs(_ty * d - _raw[0]))
target  = (_ty * _snap, _raw[1])
print(f"Target: {target}")


# ─── Digital solver ────────────────────────────────────────────────────────────
class DigitalSolver:
    @staticmethod
    def raydata(dir):
        t = 1 if dir == 0 else min(0.5 / abs(dir), 1)
        return np.linspace(0, t, 50, True)

    @staticmethod
    def solve_ray(directions):
        hit_idx, hit_dir = [], []
        for i, d in enumerate(directions):
            ty = target[1] + 0.5
            if abs(ty * d - target[0]) <= radius:
                hit_idx.append(i); hit_dir.append(d)
        return hit_idx, hit_dir

    @staticmethod
    def show_demo():
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.set_xlim((-0.5 - world_border, 0.5 + world_border))
        ax.set_ylim((-0.5 - world_border, 0.5 + world_border))
        ax.set_title("Digital Ray Solver")
        ax.add_patch(patches.Rectangle((-0.5, -0.5), 1, 1,
                                        linewidth=2, edgecolor='r', facecolor='none'))
        ax.add_patch(patches.Circle(target, radius, color='blue'))
        _, hit_dirs = DigitalSolver.solve_ray(rays)
        for d in rays:
            data  = DigitalSolver.raydata(d)
            color = 'purple' if d in hit_dirs else 'green'
            ax.plot(d * data, data - 0.5, c=color)
        plt.tight_layout(); plt.show()


# ─── Quantum arithmetic oracle ─────────────────────────────────────────────────
#
# Hit condition (same maths as DigitalSolver):
#   |ty * dir_i  -  target_x| <= radius
#   where  ty     = (target_y + 0.5)
#          dir_i  = i/(N-1) - 0.5
#
# In exact integer arithmetic (multiply through, no approximation):
#   A(i) = 2*TY*i  -  TY*(N-1)          (scaled direction * scaled ty)
#   Hit  iff   LOWER <= A(i) <= UPPER
#   where  TY    = round((target_y + 0.5) * SCALE)
#          TX    = round(target_x * SCALE)
#          LOWER = 2*(N-1)*TX - round(2*(N-1)*SCALE*radius)  + TY*(N-1)
#          UPPER = LOWER + 2*round(2*(N-1)*SCALE*radius)
#
# Because A(i) = C1*i  (with C1 = 2*TY, a constant), the circuit only needs to:
#   1. Compute  acc = C1 * i   via 4 controlled-additions (one per input qubit bit)
#   2. Subtract LOWER from acc; read MSB (1 = underflow = i < LOWER)
#   3. Subtract UPPER+1 from acc; read MSB (1 = i <= UPPER)
#   4. Phase-kickback when both conditions hold: no underflow at step 2 AND underflow at step 3
#   5. Uncompute everything to leave ancilla clean
#
# Arithmetic is implemented with Draper's QFT-based constant adder (no carry ancilla).
# Simulation uses AerSimulator with method='matrix_product_state' for speed.

class QuantumSolver:

    # ── Fixed-point constants (computed once per target) ──────────────────────
    @staticmethod
    def _constants():
        TY  = round((target[1] + 0.5) * SCALE)
        TX  = round(target[0] * SCALE)
        C1  = 2 * TY                            # multiply factor:  acc = C1 * i
        off = TY * (N - 1)                      # offset term
        B   = 2 * (N - 1) * TX                  # scaled target_x contribution
        R   = round(2 * (N - 1) * SCALE * radius)
        LOWER = B - R + off
        UPPER = B + R + off
        return C1, LOWER, UPPER

    @staticmethod
    def _hit_indices(C1, LOWER, UPPER):
        return [i for i in range(N) if LOWER <= C1 * i <= UPPER]

    # ── QFT-based constant adder ───────────────────────────────────────────────
    @staticmethod
    def _qft(n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        for j in range(n - 1, -1, -1):
            qc.h(j)
            for k in range(j - 1, -1, -1):
                qc.cp(math.pi / 2 ** (j - k), k, j)
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        return qc

    @staticmethod
    def _add_const(n: int, c: int) -> QuantumCircuit:
        """
        |x⟩  →  |x + c  mod 2^n⟩
        Uses Draper's QFT adder. No ancilla required.
        """
        qc  = QuantumCircuit(n)
        cm  = c % (1 << n)
        qc.compose(QuantumSolver._qft(n), inplace=True)
        for i in range(n):
            qc.p(2 * math.pi * cm / 2 ** (n - i), i)
        qc.compose(QuantumSolver._qft(n).inverse(), inplace=True)
        return qc

    @staticmethod
    def _ctrl_add_const(n: int, c: int) -> QuantumCircuit:
        """
        Controlled version: qubits [0..n-1] = acc, qubit n = ctrl.
        If ctrl=1: acc += c  (mod 2^n).
        """
        qc  = QuantumCircuit(n + 1)
        acc = list(range(n))
        cm  = c % (1 << n)
        qc.compose(QuantumSolver._qft(n), qubits=acc, inplace=True)
        for i in range(n):
            qc.cp(2 * math.pi * cm / 2 ** (n - i), n, acc[i])
        qc.compose(QuantumSolver._qft(n).inverse(), qubits=acc, inplace=True)
        return qc

    # ── Oracle ────────────────────────────────────────────────────────────────
    @staticmethod
    def Oracle(qc: QuantumCircuit,
               q_in, q_acc, q_msb1, q_msb2, q_ph,
               n_acc, C1, LOWER, UPPER):
        """
        Arithmetic oracle: phase-kicks |i⟩ iff LOWER <= C1*i <= UPPER.

        Register layout (all in qc):
          q_in  [0..3]        – search register (ray index i)
          q_acc [4..4+n_acc-1]– accumulator (workspace, starts and ends at 0)
          q_msb1, q_msb2      – 1-qubit ancilla for borrow flags (start/end 0)
          q_ph                – phase-kickback qubit (stays in |−⟩)

        Steps:
          1.  acc ← C1 * i  (4 controlled additions, one per input bit)
          2.  acc ← acc − LOWER;  copy MSB → msb1;  restore acc
          3.  acc ← acc − (UPPER+1);  copy MSB → msb2;  restore acc
          4.  Phase-kick q_ph when msb1=0 AND msb2=1
                (= C1*i >= LOWER  AND  C1*i <= UPPER)
          5.  Uncompute steps 3, 2, 1 in reverse to leave ancilla clean
        """
        n  = n_acc
        mod = 1 << n

        # helpers: short names
        def add(c):   return QuantumSolver._add_const(n, c)
        def cadd(c):  return QuantumSolver._ctrl_add_const(n, c)

        # Step 1: accumulate  acc = C1 * i
        for k in range(len(q_in)):
            sub = cadd(C1 * (2 ** k))
            qc.compose(sub, qubits=q_acc + [q_in[k]], inplace=True)

        # Step 2: subtract LOWER, capture borrow, restore
        qc.compose(add(mod - LOWER), qubits=q_acc, inplace=True)
        qc.cx(q_acc[-1], q_msb1)                 # msb1 = 1 if acc < LOWER
        qc.compose(add(LOWER), qubits=q_acc, inplace=True)

        # Step 3: subtract UPPER+1, capture borrow, restore
        qc.compose(add(mod - (UPPER + 1)), qubits=q_acc, inplace=True)
        qc.cx(q_acc[-1], q_msb2)                 # msb2 = 1 if acc <= UPPER
        qc.compose(add(UPPER + 1), qubits=q_acc, inplace=True)

        # Step 4: phase-kick when (msb1=0) AND (msb2=1)
        #   Flip msb1 so condition becomes msb1=1 AND msb2=1, then Toffoli
        qc.x(q_msb1)
        qc.ccx(q_msb1, q_msb2, q_ph)
        qc.x(q_msb1)

        # Step 5: uncompute msb2, msb1, acc (exact reverse of steps 3, 2, 1)
        qc.compose(add(mod - (UPPER + 1)), qubits=q_acc, inplace=True)
        qc.cx(q_acc[-1], q_msb2)
        qc.compose(add(UPPER + 1), qubits=q_acc, inplace=True)

        qc.compose(add(mod - LOWER), qubits=q_acc, inplace=True)
        qc.cx(q_acc[-1], q_msb1)
        qc.compose(add(LOWER), qubits=q_acc, inplace=True)

        for k in range(len(q_in) - 1, -1, -1):
            sub = cadd(mod - C1 * (2 ** k))
            qc.compose(sub, qubits=q_acc + [q_in[k]], inplace=True)

        qc.barrier()

    # ── Diffuser ──────────────────────────────────────────────────────────────
    @staticmethod
    def Diffuse(qc: QuantumCircuit, q_in):
        """Grover diffuser (inversion about average) on q_in only."""
        qc.h(q_in); qc.x(q_in)
        qc.h(q_in[-1])
        qc.mcx(q_in[:-1], q_in[-1])
        qc.h(q_in[-1])
        qc.x(q_in); qc.h(q_in)
        qc.barrier()

    # ── Full Grover search ────────────────────────────────────────────────────
    @staticmethod
    def Grover():
        C1, LOWER, UPPER = QuantumSolver._constants()
        hits = QuantumSolver._hit_indices(C1, LOWER, UPPER)
        M    = len(hits)

        if M == 0:
            print("No hits — Grover has nothing to search for."); return

        theta     = math.asin(math.sqrt(M / N))
        num_iters = max(1, round(math.pi / (4 * theta) - 0.5))
        print(f"Hits (classical): {hits}  |  M={M}  |  Grover iters: {num_iters}")

        # ── Qubit layout ──────────────────────────────────────────────────────
        # n_acc must satisfy:
        #   • 2^(n_acc-1) > UPPER   (so subtraction of UPPER+1 from a value
        #                             in [0, UPPER] produces MSB=1 correctly)
        #   • UPPER < 2^n_acc
        # Since UPPER = C1*(N-1) at most, we need ceil(log2(C1*(N-1)+1))+1 bits.
        n_acc  = math.ceil(math.log2(max(UPPER, C1 * (N - 1)) + 2)) + 1
        q_in   = list(range(num_qbits))
        q_acc  = list(range(num_qbits, num_qbits + n_acc))
        q_msb1 = num_qbits + n_acc
        q_msb2 = num_qbits + n_acc + 1
        q_ph   = num_qbits + n_acc + 2
        total  = num_qbits + n_acc + 3
        print(f"n_acc={n_acc}, total qubits={total}")

        qc = QuantumCircuit(total, num_qbits)

        # Uniform superposition on search register
        qc.h(q_in)
        # Phase-kickback ancilla in |−⟩
        qc.x(q_ph); qc.h(q_ph)
        qc.barrier()

        for _ in range(num_iters):
            QuantumSolver.Oracle(qc, q_in, q_acc, q_msb1, q_msb2, q_ph,
                                  n_acc, C1, LOWER, UPPER)
            QuantumSolver.Diffuse(qc, q_in)

        qc.measure(q_in, range(num_qbits))

        # MPS simulator: handles deep circuits on 20+ qubits efficiently
        sim    = AerSimulator(method='matrix_product_state')
        tqc    = transpile(qc, sim)
        counts = sim.run(tqc, shots=2048).result().get_counts()

        # Qiskit bitstrings are MSB-first → int(k, 2) directly gives the index
        total_shots = sum(counts.values())
        index_probs = {int(k, 2): v / total_shots for k, v in counts.items()}

        top = max(index_probs, key=index_probs.get)
        print(f"Top measured index: {top}  (prob={index_probs[top]:.3f})  "
              f"correct: {top in hits}")

        QuantumSolver._show_results(index_probs, hits, qc)

    # ── Visualise ─────────────────────────────────────────────────────────────
    @staticmethod
    def _show_results(index_probs, hits, qc):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Grover's Algorithm — Arithmetic Quantum Ray Solver", fontsize=14)

        # Ray plot
        ax = axes[0]
        ax.set_aspect(1)
        ax.set_xlim((-0.5 - world_border, 0.5 + world_border))
        ax.set_ylim((-0.5 - world_border, 0.5 + world_border))
        ax.set_title("Ray Visualisation")
        ax.add_patch(patches.Rectangle((-0.5, -0.5), 1, 1,
                                        linewidth=2, edgecolor='r', facecolor='none'))
        ax.add_patch(patches.Circle(target, radius, color='blue'))
        top_idx = max(index_probs, key=index_probs.get)
        from matplotlib.lines import Line2D
        for i, d in enumerate(rays):
            data = np.linspace(0, 1, 50)
            if   i in hits:     ax.plot(d * data, data - 0.5, c='purple', lw=2.5)
            elif i == top_idx:  ax.plot(d * data, data - 0.5, c='orange', lw=2)
            else:               ax.plot(d * data, data - 0.5, c='green',  alpha=0.4, lw=0.8)
        ax.legend(handles=[
            Line2D([0],[0], color='purple', lw=2, label='true hit'),
            Line2D([0],[0], color='orange', lw=2, label='quantum top'),
            Line2D([0],[0], color='green',  lw=1, label='miss'),
        ], fontsize=8)

        # Probability histogram
        ax2 = axes[1]
        bar_probs  = [index_probs.get(i, 0) for i in range(N)]
        bar_colors = ['purple' if i in hits else 'steelblue' for i in range(N)]
        ax2.bar(range(N), bar_probs, color=bar_colors)
        ax2.axhline(1 / N, color='red', linestyle='--', label='uniform (1/N)')
        ax2.set_xlabel("Ray index"); ax2.set_ylabel("Probability")
        ax2.set_title("Measured Distribution\n(purple = true hit)"); ax2.legend()

        # Circuit diagram
        ax3 = axes[2]; ax3.axis('off')
        cfig = qc.draw('mpl', fold=40)
        cfig.savefig('/tmp/circuit.png', bbox_inches='tight', dpi=72)
        plt.close(cfig)
        ax3.imshow(plt.imread('/tmp/circuit.png'))
        ax3.set_title("Grover Circuit")

        plt.tight_layout(); plt.show()


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DigitalSolver.show_demo()
    QuantumSolver.Grover()
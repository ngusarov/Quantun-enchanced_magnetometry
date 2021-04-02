import math
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)


def randbin(data, F):
    phi = data.const * F * data.t
    p_0 = (math.sin(phi)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]


simulator = Aer.get_backend('qasm_simulator')

def randbin2(data, F):
    phi = data.const * F * data.t

    # Use Aer's qasm_simulator


    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(1)

    circuit.h(0)
    circuit.rz(phi, 0)
    circuit.h(0)

    # Map the quantum measurement to the classical bits
    circuit.measure_all()

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return int(list(counts.keys())[0])
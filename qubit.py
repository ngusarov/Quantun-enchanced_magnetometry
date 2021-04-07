import math
import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)

from qiskit import IBMQ


def randbin(data, F):
    phi = data.const * F * data.t

    p_0 = (math.sin(phi)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]



provider = IBMQ.enable_account(token=token)
backend = provider.get_backend("ibmq_armonk")
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')
def randbin2(data, F):
    phi = data.const * F * data.t

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q)
    circuit.u3(0, 0, math.pi - 2*phi, q)
    circuit.h(q)

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, backend=backend, shots=1)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return int(list(counts.keys())[0])

'''
def randbin3(data, F):
    phi = data.const * F * data.t

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(1)

    circuit.h(0)
    circuit.u1(phi, 0)
    circuit.h(0)

    # Map the quantum measurement to the classical bits
    circuit.measure(0)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=1000)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return counts
    
'''
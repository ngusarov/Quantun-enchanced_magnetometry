import math
from qiskit.providers.ibmq import least_busy
import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)

from qiskit import IBMQ


def randbin(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree

    p_0 = (math.sin(phi/2)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]

#IBMQ.disable_account()
token ='d450d58f70726aa812595264cebdcc1b954e95cde187217ab4cbe3be5c27a3d330fb6a8fd34007762796f423d2fd7078952738e351a7828397cc184e48d86a6e'
provider = IBMQ.enable_account(token)
#backend = provider.get_backend("ibmq_armonk")
backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator))
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')


def randbin2(data, F): #real machine
    phi = data.const * F * data.t * data.F_degree

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q)
    circuit.rz(phi, q)
    circuit.h(q)

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, backend=backend, shots=5000)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print(counts, 'for real')

    return int(counts['0'])/5000


def randbin3(data, F): # simulator --- WORKS with reversed probabilities
    phi = data.const * F * data.t * data.F_degree

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q)
    circuit.rz(phi, q)
    circuit.h(q)

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, shots=10000)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print(counts, 'for sim')

    return int(counts['0'])/10000 #int(list(counts.keys())[0])
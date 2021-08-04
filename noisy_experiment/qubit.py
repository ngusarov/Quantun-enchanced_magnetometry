import math

import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)

import qiskit.providers.aer.noise as noise

from qiskit import IBMQ



def randbin(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree

    p_0 = (math.sin(phi)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]


#
#provider = IBMQ.enable_account()
#backend = provider.get_backend("ibmq_armonk")
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

'''
def randbin2(data, F): #real machine
    phi = data.const * F * data.t * data.F_degree

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q)
    circuit.rz(math.pi - 2*phi, q)
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

def randbin3(data, F, theta_sphere, phi_sphere=0): # simulator --- WORKS
    prob_1 = 0.3
    error_1 = noise.depolarizing_error(prob_1, 1)
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u3'])
    basis_gates = noise_model.basis_gates


    phi = data.const * F * data.t * data.F_degree

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    rotate_angle = math.pi - 2*phi
    circuit.u3(theta_sphere, phi_sphere, 0, q)
    circuit.rz(2*phi, q)
    circuit.h(q)

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, Aer.get_backend('qasm_simulator'),
                 basis_gates=basis_gates,
                 noise_model=noise_model, shots=1)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return int(list(counts.keys())[0])

import math
#from qiskit.providers.ibmq import least_busy
import numpy as np
#from qiskit import (
#    QuantumCircuit,
#    execute,
#    Aer, QuantumRegister, ClassicalRegister)

#from qiskit import IBMQ

def W_CP(f, a, n, t):
    PI = math.pi
    tau = t / n
    return np.sin(PI*f*n*tau)*(1-1/np.cos(PI*f*tau))*np.cos(PI*f*n*tau + a) / (PI*f*n*tau)


def randbin(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree * W_CP(data.f_ac, data.phase, data.n_of_pulses, data.t)

    p_0 = (math.sin(phi/2)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]
'''
#IBMQ.disable_account()

provider = IBMQ.enable_account(t)
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


def randbin3(data, F): # simulator --- WORKS
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
    job = execute(circuit, simulator, shots=200)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print(counts)

    return int(counts['0'])/200 #int(list(counts.keys())[0])
'''
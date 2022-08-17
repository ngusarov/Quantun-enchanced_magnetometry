import math
import numpy as np

from qiskit.providers.ibmq import least_busy
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)
import qiskit.providers.aer.noise as noise
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ

#
# Мой код, который я когда-то начинал писать для AC
#

def W_CP(f, a, n, t): # a way of finding qubit phi via Carr-Purcell formula from Degen-2017
    # TODO check for errors
    PI = math.pi
    tau = t / n
    return np.sin(PI*f*n*tau)*(1-1/np.cos(PI*f*tau))*np.cos(PI*f*n*tau + a) / (PI*f*n*tau)



def math_qubit(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree * W_CP(data.f_ac, data.phase, data.n_of_pulses, data.t)

    p_0 = (math.sin(phi/2)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]

#''' Use for qiskit real qubits
#IBMQ.disable_account() # in case qiskit breaks down uncomment, run, then comment back
token = '' # there should be your token from https://quantum-computing.ibm.com/login
provider = IBMQ.enable_account(token)

# two following variants for real qubits
#backend = provider.get_backend("ibmq_armonk") # particular device -- the only with pulses!!!
backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator)) # least busy REAL device
#'''

def real_qubit(data, F): #real machine
    phi = data.const * F * data.t * data.F_degree * W_CP(data.f_ac, data.phase, data.n_of_pulses, data.t)

    q = QuantumRegister(1) # to sense magnetic field
    c = ClassicalRegister(1) # ancilla to take measurements

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q) # hadamard
    circuit.rz(phi, q) # apply particular equatorial rotation
    circuit.h(q) # hadamard

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    N_qubits = 100
    # Execute the circuit on the qasm simulator
    job = execute(circuit, backend=backend, shots=N_qubits) # it will repeat experiment for SAME 'phi' 'shots' times -- exact assemble sensing
    job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit) # counts is a dictionary

    counts = {["1"]: 100}


    return ??? # returns most frequent result, for assemble sensing of AC replace with the following

    # returns probability
    '''
    try:
        return counts['0'] / sum(counts.values()) # p_0 or p_1 -- needs to be checked; Qiskit may reverce p_0 and p_1
    except Exception: # no '0' key was found
        return 0 # p_0 == 0 ; OR 'return 1'!!!!!
    '''


# Use Aer's qubit simulator
simulator = Aer.get_backend('qasm_simulator')

def simulator_qubit(data, F): # simulator
    phi = data.const * F * data.t * data.F_degree* W_CP(data.f_ac, data.phase, data.n_of_pulses, data.t)

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
    job = execute(circuit, simulator, shots=data.n_rep)
    job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)

    return int(list(counts.keys())[0])  # returns most frequent result, for assemble sensing of AC replace with the following

    # returns probability
    '''
    try:
        return counts['0'] / sum(counts.values()) # p_0 or p_1 -- needs to be checked; Qiskit may reverce p_0 and p_1
    except Exception: # no '0' key was found
        return 0 # p_0 == 0 ; OR 'return 1'!!!!!
    '''
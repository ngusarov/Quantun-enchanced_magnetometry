import math
from qiskit.providers.ibmq import least_busy
import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister, assemble)
import qiskit.providers.aer.noise as noise
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit import IBMQ
from qiskit.tools import job_monitor


def randbin(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree

    p_0 = (math.sin(phi/2)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]

#IBMQ.disable_account()

#backend = provider.get_backend("ibmq_armonk")
#backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator))
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

'''
def randbin2(data, F): #real machine
    N = int(math.log(200*10**(-6)/data.t_init)/math.log(2))

    phi = data.const * F * data.t_init * data.F_degree

    q = QuantumRegister(N)
    c = ClassicalRegister(N)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    for i in range(N):
        circuit.h(i)
        circuit.rz(phi*2**i, i)
        circuit.h(i)

        # Map the quantum measurement to the classical bits
        circuit.measure(i, i)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, backend=backend, shots=data.num_of_repetitions)
    job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print(counts, 'for real')

    answers = {}
    for i in range(N):
        zeros = 0
        ones = 0
        for each in list(counts.keys()):
            if each[i] == '0':
                zeros += 1
            elif each[i] == '1':
                ones += 1
        if zeros > ones:
            answers[data.t_init*2**(i)] = 1
        else:
            answers[data.t_init*2**(i)] = 0
    return answers

    
'''

def randbin3(data, F): # simulator --- WORKS with reversed probabilities
    #error_1 = noise.phase_amplitude_damping_error(param_phase=1, param_amp=0)
    print(data.t*10**6)
    error_2 = noise.thermal_relaxation_error(0.001, 0.001, data.t*10**6, excited_state_population=0)
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_2, ['rz'])
    basis_gates = noise_model.basis_gates

    phi = data.const * F * data.t * data.F_degree

    q = QuantumRegister(1)
    c = ClassicalRegister(1)

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(q, c)

    circuit.h(q)
    #circuit.ry(math.pi/2+delta, q)
    #circuit.u3(math.pi / 2, 0, math.pi, q)
    #circuit.u3(-math.pi / 4, 0, 0, q)
    circuit.rz(phi, q)
    circuit.h(q)

    # Map the quantum measurement to the classical bits
    circuit.measure(q, c)

    # Execute the circuit on the qasm simulator
    job = execute(circuit, simulator, basis_gates=basis_gates,
                 noise_model=noise_model, shots=1000)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    #print(counts, 'for sim')
    try:
        return int(counts['0'])/1000 #int(list(counts.keys())[0])
    except Exception:
        return 0
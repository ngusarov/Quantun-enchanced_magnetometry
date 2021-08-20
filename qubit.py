import math
from qiskit.providers.ibmq import least_busy
import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)
import qiskit.providers.aer.noise as noise
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor


def randbin(data, F): # simple math
    phi = data.const * F * data.t * data.F_degree
    p_0 = (math.sin(phi/2)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]

#IBMQ.disable_account()

backend = provider.get_backend("ibmq_manila")
#backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator))
# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

#'''
def randbin2(data, F): #real machine
    N = int(math.log(200*10**(-6)/data.t_init)/math.log(2))
    N_qub = 5

    answers = {}

    phi = data.const * F * data.t_init * data.F_degree
    n_rep = 0
    while n_rep*N_qub < N:
        q = QuantumRegister(N_qub)
        c = ClassicalRegister(N_qub)

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(q, c)

        for i in range(N_qub):

            if i+n_rep*N_qub > N:
                break

            circuit.h(i)
            circuit.rz(phi*2**(i+n_rep*N_qub), i)
            circuit.h(i)

            print((math.cos(phi*2**(i+n_rep*N_qub)/2))**2)

            # Map the quantum measurement to the classical bits
            circuit.measure(i, i)

        # Execute the circuit on the qasm simulator
        job = execute(circuit, backend=backend, shots=5001)
        job_monitor(job)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(circuit)
        print(counts, 'for real')
        print('t_init ', data.t_init)


        for i in range(N_qub):

            if i+n_rep*N_qub > N:
                break

            zeros = 0
            ones = 0
            for each in list(counts.keys()):
                if each[-i-1] == '0':
                    zeros += counts[each]
                elif each[-i-1] == '1':
                    ones += counts[each]
            if zeros > ones:
                answers[data.t_init*2**(i+n_rep*N_qub)] = 1
            else:
                answers[data.t_init*2**(i+n_rep*N_qub)] = 0

        n_rep += 1

    print(answers)
    return answers

    # return counts
    '''
    try:
        if counts['1'] < counts['0']:  # LOWEST BECAUSE PROBS ARE REVERSED!!!
            return 1
        else:
            return 0
    except Exception:
        return abs(int(list(counts.keys())[0]) - 1)
    '''
# '''

def randbin3(data, F): # simulator --- WORKS with reversed probabilities
    #error_1 = noise.depolarizing_error(data.phase_err, 1)
    error_1 = noise.phase_amplitude_damping_error(param_phase=data.phase_err, param_amp=data.amp_err)
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['rz'])
    basis_gates = noise_model.basis_gates

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
    job = execute(circuit, simulator, basis_gates=basis_gates,
                 noise_model=noise_model,shots=data.num_of_repetitions)
    #job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts(circuit)
    print(counts, 'for sim')

    #return counts
    #'''
    try:
        if counts['1'] < counts['0']: # LOWEST BECAUSE PROBS ARE REVERSED!!!
            return 1
        else:
            return 0
    except Exception:
        return abs(int(list(counts.keys())[0])-1)
    #'''
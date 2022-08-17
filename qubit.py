import math
import random
import time

from qiskit.providers.ibmq import least_busy
import numpy as np
from qiskit import (
    QuantumCircuit,
    execute,
    Aer, QuantumRegister, ClassicalRegister)
import qiskit.providers.aer.noise as noise
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from tqdm import tqdm


def randbin(data, F, i): # simple math
    phi = data.const * F * data.t * data.F_degree
    p_0 = (math.sin(phi/2)) ** 2

    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]
'''
try:
    IBMQ.disable_account()
except Exception:
    pass
token = '1e59d98e02c0540ac85a62e0aeb450bb17e26bc72e68471cfc0c26a8b9fa3b3f6d20c55f26866c5a415203278e8fdae17b65aa9b798243da76b2d3b45add1193'
provider = IBMQ.enable_account(token=token)
backend = provider.get_backend("ibmq_manila")
#backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator))
# Use Aer's qasm_simulator
#'''
simulator = Aer.get_backend('qasm_simulator')


'''
def randbin2(data, F): #real machine
    N = int(math.log(data.T_2*10**(-6)/data.t_init)/math.log(2))
    N_qub = 1

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
        job = execute(circuit, backend=backend, shots=data.num_of_repetitions)
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
    
    
    try:
        if counts['1'] < counts['0']:  # LOWEST BECAUSE PROBS ARE REVERSED!!!
            return 1
        else:
            return 0
    except Exception:
        return abs(int(list(counts.keys())[0]) - 1)
    

#'''

#'''
def randbin3(data, F): # simulator --- WORKS with reversed probabilities

    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    cir = QuantumCircuit(q, c)

    Answers = []
    for i in range(data.REP):
        Answers.append({})

    for i in tqdm(range(data.STEPS)):
        Circuits = [cir for _ in range(data.REP)]
        for j in range(data.REP):
            error_2 = noise.thermal_relaxation_error(data.T_1*10**(-6), data.T_2*10**(-6), data.t_init*2**i, excited_state_population=0.5)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_2, ['rz', 'h'])
            basis_gates = noise_model.basis_gates

            phi = data.const * F * data.t_init*2**i * data.F_degree

            q = QuantumRegister(1)
            c = ClassicalRegister(1)

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(q, c)

            circuit.h(q)
            circuit.delay(data.t, unit='s')
            circuit.rz(phi, q)
            circuit.h(q)

            # Map the quantum measurement to the classical bits
            circuit.measure(q, c)

            Circuits[j] = circuit

            # Execute the circuit on the qasm simulator
        job = execute(Circuits, simulator,
                        basis_gates=basis_gates,
                        noise_model=noise_model,
                        shots=data.num_of_repetitions)
        #job_monitor(job)
        result = job.result()

        for j in range(data.REP):
            # Returns counts
            counts = result.get_counts(Circuits[j])
            #print(counts)
            #print(counts, 'for sim')

            #return counts

            try:
                if counts['1'] < counts['0']: # LOWEST BECAUSE PROBS ARE REVERSED!!!
                    Answers[j][data.t_init*2**i] = 1
                else:
                    Answers[j][data.t_init*2**i] = 0
            except Exception:
                Answers[j][data.t_init*2**i] = abs(int(list(counts.keys())[0])-1)

    #print(Answers)
    return Answers
#'''

def randbin4(data, F):
    Circuits = [0 for _ in range(data.REP*data.STEPS)]

    for j in range(data.REP):
        for i in range(data.STEPS):
            phi = data.const * F * data.t_init*2**i * data.F_degree
            p_0 = ((math.sin(phi / 2)) ** 2 - 0.5) * np.exp(
                -data.t_init*2**i * 10 ** 6 / data.T_2) + 0.5

            Circuits[j * data.STEPS + i] = int(round(sum([np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0] for k in range(data.num_of_repetitions)])/data.num_of_repetitions))
            #Circuits[j * data.STEPS + i] = int(round(np.mean(random.choices([0, 1], weights=[p_0, 1 - p_0], k=41))))
            #time.sleep(0.03)

    Answers = []
    for j in range(data.REP):
        Answers.append({})
        for i in range(15):
            Answers[j][data.t_init * 2 ** i] = Circuits[j * data.STEPS + i]
    #print(Answers)
    return Answers

def randbin3_test(data, F): # simulator --- WORKS with reversed probabilities:

    times = np.linspace(0, 10*data.t_init, 10)

    Answers = []

    for i in range(10):
        error_2 = noise.thermal_relaxation_error(data.T_1 * 10 ** (-6), data.T_2 * 10 ** (-6), times[i],
                                                 excited_state_population=0.5)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_2, ['rz', 'h'])
        basis_gates = noise_model.basis_gates

        phi = data.const * F * times[i] * data.F_degree

        q = QuantumRegister(1)
        c = ClassicalRegister(1)

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(q, c)

        circuit.h(q)
        circuit.delay(times[i], unit='s')
        circuit.rz(phi, q)
        circuit.h(q)

        # Map the quantum measurement to the classical bits
        circuit.measure(q, c)

        # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator,
                        basis_gates=basis_gates,
                        noise_model=noise_model,
                        shots=1000)
        job_monitor(job)

        #Circuits = Circuits.reshape(data.REP, 15)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(circuit)
        print(counts)
        #print(counts, 'for sim')

        #return counts

        try:
            Answers.append(int(counts['0']) / 1000)  # int(list(counts.keys())[0])
        except Exception:
            Answers.append(0)

        print(Answers)

    print(Answers)
    return Answers

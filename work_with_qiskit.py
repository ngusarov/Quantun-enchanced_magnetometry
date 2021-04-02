import math
import experiment
from qiskit import IBMQ

#token = '6a8a275fe0b9398a9ad3ce716594d8f279cc05ca2ec5c1abf48ccb779a25cc1f16c02623501b4d1278347ead3bdcbcde73e8ad4c85b9b3441d8bfaff98a54191'
#IBMQ.save_account(token=token)


from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(1)

# Add a H gate on qubit 0
phi = experiment.ExperimentData.const * 37 * experiment.ExperimentData.t
print(phi)

circuit.h(0)
circuit.p(phi, 0)
circuit.h(0)

# Map the quantum measurement to the classical bits
circuit.measure_all()

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print(counts)

# Draw the circuit
circuit.draw()
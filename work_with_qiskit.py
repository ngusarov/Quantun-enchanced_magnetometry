import math

import qubit
import experiment

data = experiment.ExperimentData()
F = 1.25


phi = data.const * F * data.t
print(phi)
print((math.sin(phi))**2)
N = 1000
count_simple = 0  # zeros
count_simulator = 0  # zeros
for i in range(N):
    if qubit.randbin(data, F) == 0:
        count_simple += 1
    if qubit.randbin2(data, F) == 0:
        count_simulator += 1

#print(qubit.randbin3(data, F)['0']/1000)

print("simulator", count_simulator/N)
print("simple", count_simple/N)

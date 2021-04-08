import math

import qubit
import experiment

data = experiment.ExperimentData()
F = 500


phi = data.const * F * data.t
print(data.const * F * data.t)
print(data.const * data.F_min * data.t)
print(data.const * data.F_max * data.t)
print((math.sin(phi))**2)
N = 300
count_simple = 0  # simple math -- randbin
count_machine = 0  # machine 1 -- randbin2
count_machine2 = 0 # machine 2 -- randbin 4
count_simulator = 0 # simulator -- randbin3
for i in range(1, N+1):
    if qubit.randbin(data, F) == 0:
        count_simple += 1
    if qubit.randbin2(data, F) == 0:
        count_machine += 1
    if qubit.randbin4(data, F) == 0:
        count_machine2 += 1
    if qubit.randbin4(data, F) == 0:
        count_simulator += 1
    print(count_simulator/i, count_simulator/N, count_machine/i, count_machine/N, count_machine2/i, count_machine2/N)

#print(qubit.randbin3(data, F)['0']/1000)

print("rz: pi - phi", count_machine/N)
print("rz: 2*phi", count_machine2/N)
print("rz_sim: pi - 2*phi -- must work", count_simulator/N)
print("math:", count_simple/N)

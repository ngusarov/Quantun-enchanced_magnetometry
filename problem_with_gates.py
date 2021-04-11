import math

import qubit
import experiment

data = experiment.ExperimentData()
F = 700

phi = data.const * F * data.t
print(data.const * F * data.t)
print(data.const * data.F_min * data.t)
print(data.const * data.F_max * data.t)
print((math.sin(phi))**2)

count = 0
for i in range(1000):
    if qubit.randbin(data, F) == 0:
        count+=1
print("math: ", count/1000)

print("simulator: ", qubit.randbin3_(data, F)['0']/1000)
print("real machine: ", qubit.randbin2_(data, F)['0']/10000)





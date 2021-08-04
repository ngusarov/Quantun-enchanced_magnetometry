import math

import qubit
import experiment
#import ramsey_qubit

data = experiment.ExperimentData()
data.F = 8

phi = data.const * data.F * data.t * data.F_degree
print(phi)
print(data.const * data.F_min * data.t * data.F_degree)
print(data.const * data.F_max * data.t * data.F_degree)
print(data.t)
print((math.sin(phi))**2)

count = 0
count_sim = 0
count_imp = 0
N = 1000
for i in range(1, N):
    if qubit.randbin(data, data.F) == 0:
        count+=1
    print("math: ", count/i)

    print(data.t * 10 ** 7)
    #if ramsey_qubit.output(data.t * 10**6) == 0:
    #    #print(data.t * 10**7)
    #    count_imp+=1
    #print("imp: ", count_imp/i)

    if qubit.randbin3(data, data.F) == 0:
        count_sim += 1
    print("simulator: ", count_sim / i)
    print("ideal: ", (math.sin(phi))**2)
import math

import qubit
import experiment
#import ramsey_qubit
import tqdm
import numpy as np
import matplotlib.pyplot as plt

data = experiment.ExperimentData()
data.F = 10
'''
phi = data.const * data.F * data.t * data.F_degree
print(phi)
print(data.const * data.F_min * data.t * data.F_degree)
print(data.const * data.F_max * data.t * data.F_degree)
print(data.t)
print((math.sin(phi/2))**2)

data.t *= 4
phi = data.const * data.F * data.t * data.F_degree
print(phi)

count = 0
count_sim = 0
count_imp = 0
N = 1000

id = (math.cos(phi/2))**2

deltas = np.linspace(-0.75*math.pi, 0.75*math.pi, 101)
errs = np.linspace(0.9, 0.9, 1)
times = np.linspace(0, 200, 5)
Fs = np.linspace(0, 50, 11)
max_anss = []
mean_val = []
for time in times:
    #data.F = F
    phi = data.const * data.F * data.t * data.F_degree
    anss = []
    print(time)
    for delta in deltas:
        ans = (qubit.randbin3(data, data.F, 0, time, delta)) #ones
        anss.append(ans)
        #print(ans, delta, err)
    #mean_val.append(min(anss))
    #max_anss.append(max(anss))
#plt.plot(errs, max_anss)
#plt.plot(errs, 0.5*np.sqrt(1-errs))
#plt.plot(errs[:-1], [(max_anss[i+1] - max_anss[i])/(errs[i+1]-errs[i]) for i in range(len(max_anss)-1)], '-')
#plt.plot(Fs, mean_val)
    plt.plot(deltas[np.argmin(anss)], min(anss), 'x')
    plt.plot(deltas, anss, ls='-', label='{}'.format(time))
    plt.plot(deltas, [( math.cos(phi/2) )**2-(0.5*np.cos(phi) + 0.5*np.exp(-time/100)*np.sin(delta + phi - math.pi/2)) for delta in deltas], label='fit {}'.format(time))
plt.legend(loc='best')
plt.show()
plt.close()

#print("simulator: ", 1-qubit.randbin3(data, data.F))
#print('real machine', 1-qubit.randbin2(data, data.F))

for i in range(1, N):
    if qubit.randbin(data, data.F) == 0:
        count+=1


    #print(data.t * 10 ** 7)
    #if ramsey_qubit.output(data.t * 10**6) == 0:
    #    #print(data.t * 10**7)
    #    count_imp+=1
    #print("imp: ", count_imp/i)
print("math: ", count / N)
print("ideal: ", (math.sin(phi/2))**2)
print("sim+real counted for 1, others for 0")
'''
ts = np.linspace(0, 100, 100)
ideals = []
sims = []
for t in ts:
    data.t = t
    phi = data.const*data.F*data.F_degree*data.t
    ideals.append(((math.cos(phi/2))**2-0.5)* np.exp(-data.t/120)+0.5)
    sims.append(qubit.randbin3(data, data.F, 0, 0, 0))

plt.plot(ts, ideals)
plt.plot(ts, sims)

plt.show()
plt.close()

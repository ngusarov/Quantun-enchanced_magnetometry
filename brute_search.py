import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import bayesians_learning
import plotter
#import ramsey_qubit

# constants start ---------------------
import qubit

def gaussian(sigma, center, x):
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.exp(-(x - center) ** 2 / (2 * sigma ** 2))


@dataclass
class ExperimentData:
    F = 20
    F_min = 0  # min field Tesla
    F_max = 50 # max field Tesla
    F_degree = 10**(-9)

    time_const = 2
    mu = 10 ** (5) * 927 * 10**(-26)  # magnetic moment of the qubit
    h = 6.62 * 10 ** (-34)  # plank's constant
    const = mu/h  # mu/h
    t = math.pi/(const*F_degree*F_max/2)*2**(-1)
    t_init = t # time of interaction in seconds
    num_of_repetitions = 101  # repetitions for one experiment

    #probability_distribution = [1 / fields_number] * fields_number


# constants end -----------------------

# work with distribution -----------


def perform():
    experimentData = ExperimentData()
    sigma = {}
    a_from_t_sum = {} #sensitivity
    a_from_step = {} #sensitivity
    N = 40
    t_sum = 0
    epsilon = 10 ** (-5)



    for step in range(N):
        outcome = int(round( sum([qubit.randbin(experimentData, experimentData.F)
                                  for i in range(experimentData.num_of_repetitions)])/
                                    experimentData.num_of_repetitions ))

        if bayesians_learning.P_qubit_state_on_F_i(outcome, experimentData.F_min, experimentData)\
                > bayesians_learning.P_qubit_state_on_F_i(outcome, experimentData.F_max, experimentData):
            experimentData.F_max = (experimentData.F_max + experimentData.F_min)/2
        else:
            experimentData.F_min = (experimentData.F_max + experimentData.F_min) / 2

        t_sum += experimentData.t * experimentData.num_of_repetitions

        current_sigma = (experimentData.F_max - experimentData.F_min)/2

        sigma[t_sum] = current_sigma

        center = (experimentData.F_max + experimentData.F_min) / 2
        a_from_t_sum[experimentData.t] = max(abs(center - experimentData.F), current_sigma) * (t_sum) ** 0.5
        #a_from_step[step] = current_sigma * (t_sum) ** 0.5

        experimentData.t *= experimentData.time_const

        print(step, center, current_sigma, experimentData.t*10**6, t_sum)
        #x = np.arange(experimentData.F_min, experimentData.F_max, 0.01)

        #plt.plot(x, 1 / (current_sigma * np.sqrt(2 * np.pi)) *
        #         np.exp(- (x - center) ** 2 / (2 * current_sigma ** 2)),
        #         linewidth=2, color='r')




        if current_sigma <= epsilon or experimentData.t >= 200*10**(-6):
            break

    #plt.show()
    #plt.close()

    print(list(sigma.keys())[-1], list(sigma.values())[-1])

    #try:
    #    plotter.plotting_sensitivity(a_from_step, r'$N$')
    #except Exception:
    #    pass
    try:
        plotter.plotting_sensitivity(a_from_t_sum, r'$t_{coherense\_max}, \, \mu s$')
    except Exception:
        pass

    plotter.plotting(sigma)

    return a_from_t_sum


'''def average_20(Field, F_max):
    F = []
    K = []
    Time = []
    Sigma = []
    counter = 0
    for i in range(1):
        try:
            print(experimentData.F_max)
            f, k, time, sigma = perform(Field)
            F.append(f)
            K.append(k)
            Time.append(time)
            Sigma.append(sigma)
            counter += 1
        except Exception as e:
            pass

    with open('experiment.txt', 'a') as f:
        f.write(str(F_max)+' '+str(Field) + ' '+ str(sum(F)/counter) + ' ' + str(sum(K)/counter)+' ' +str(sum(Time)/counter) + ' ' + str(sum(Sigma)/counter)+'\n')
'''

if __name__ == "__main__":
    perform()
    #for i in range(200):
    #    average_20(50, experimentData.F_max)
    #    experimentData.F_max += 30
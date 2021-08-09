from dataclasses import dataclass
import matplotlib.pyplot as plt

import bayesians_learning
import plotter
#import ramsey_qubit

# constants start ---------------------
#F = 20  # field strength to be measured in Tesla


@dataclass
class ExperimentData:
    F = 40
    F_min = 0  # min field Tesla
    F_max = 100 # max field Tesla
    F_degree = 10**(-9)
    gained_degree = 1
    delta_F = 1  # accuracy of F defining
    fields_number = round( (F_max - F_min + delta_F) / delta_F ) # amount of discrete F meanings
    time_const = 2
    mu = 10 ** (5) * 927 * 10**(-26)  # magnetic moment of the qubit
    h = 6.62 * 10 ** (-34)  # plank's constant
    const = mu/h  # mu/h
    reverse = False
    t = 3.14/4/(0.5*const*F_degree*(F_max - F_min)/2) * 2**(1)  # time of interaction in seconds
    num_of_repetitions = 2  # repetitions for one experiment

    probability_distribution = [1 / fields_number] * fields_number


# constants end -----------------------

# work with distribution -----------


def find_peak(data):
    """
    finds peak
    :param data: experiment data
    :return: x, y(x) # peak
    """
    y_max = data.probability_distribution[0]
    x_max = data.F_min
    for i in range(1, data.fields_number):
        if data.probability_distribution[i] > y_max:
            x_max = data.F_min + data.delta_F*i
            y_max = data.probability_distribution[i]
    return x_max, y_max


def localise_peak(data):
    left_ind = 0
    right_ind = data.fields_number-1
    for i in range(1, data.fields_number-1):
        if (data.probability_distribution[i+1] - data.probability_distribution[i])*0.5 > \
                data.probability_distribution[i] - data.probability_distribution[i-1]:
            left_ind = i-1
            break

    for i in range(data.fields_number-2, 0, -1):
        if (data.probability_distribution[i-1] - data.probability_distribution[i])*0.5 > \
                data.probability_distribution[i] - data.probability_distribution[i+1]:
            right_ind = i+1
            break
    return left_ind, right_ind





def find_sigma(x_peak, y_peak, data):
    """
    :param x_peak:
    :param y_peak:
    :param data:
    :return: sigma
    """
    y_sigma = y_peak / (2.81**0.5)
    epsilon = max([ abs( data.probability_distribution[i] - data.probability_distribution[i-1] )/2 for i in range(1, data.fields_number) ])
    x_sigma = []
    for i in range(1, data.fields_number):
        if abs(data.probability_distribution[i] - y_sigma) <= epsilon:
            x_sigma.append(data.F_min + data.delta_F*i)
    if len(x_sigma) > 0:
        return min([ abs(x_sigma[i] - x_peak) for i in range(len(x_sigma)) ])
    else:
        return 0

def find_sigma_2(x_peak, y_peak, data):
    """
    :param x_peak:
    :param y_peak:
    :param data:
    :return: sigma
    """
    return 1/((2*3.1415)**(0.5)*y_peak)
# work with distribution -----------

# enlarging field segment ----------
def expand(x_peak, y_peak, sigma, data):
    en_param = 20

    new_F_min = (x_peak - (en_param//2)*data.delta_F)*10
    start_ind = int(round((new_F_min//10 - data.F_min) / data.delta_F))
    data.F_min = new_F_min + 1
    data.F_max = (x_peak + (en_param//2)*data.delta_F)*10
    data.F_degree /= 10
    data.F *= 10
    data.gained_degree *= 10
    data.fields_number = int(round( (data.F_max - data.F_min + data.delta_F) / data.delta_F ))
    new_distr = [0] * data.fields_number

    for i in range(start_ind+1, start_ind + en_param + 1):
        new_distr[(i - start_ind)*10 - 1] = data.probability_distribution[i]

    for i in range(start_ind, start_ind + en_param):
        for j in range(1, 10):
            new_distr[(i - start_ind) * 10 + j - 1] = j/10*(data.probability_distribution[i+1] - data.probability_distribution[i]) + \
                data.probability_distribution[i]
    data.probability_distribution = new_distr
    '''
    integral = bayesians_learning.integrate_distribution(data)
    print(integral)
    k = 1 / integral
    data.probability_distribution = list(map(lambda x: k*x, data.probability_distribution))
    '''

# enlarging field segment ----------


def perform():
    experimentData = ExperimentData()
    sigma = {}
    a_from_t_sum = {} #sensitivity
    a_from_step = {} #sensitivity
    N = 25
    t_sum = 0
    epsilon = 10 ** (-3)
    prev_sigma = experimentData.F_max - experimentData.F_min
    flag = False
    prev_step = 0
    #
    fig, ax = plt.subplots()
    ax.minorticks_on()

    print(experimentData.probability_distribution) # initial
    print(experimentData.fields_number)
    for step in range(N):

        bayesians_learning.renew_probalities(experimentData)
        #bayesians_learning.renew_probalities(qubit.randbin3(experimentData, F), experimentData)
        #bayesians_learning.renew_probalities(qubit.randbin2(experimentData, F), experimentData)
        #bayesians_learning.renew_probalities(ramsey_qubit.output(experimentData.t), experimentData)
        t_sum += experimentData.t * experimentData.num_of_repetitions



        x_peak, y_peak = find_peak(experimentData)
        left_peak_side, right_peak_side = localise_peak(experimentData)
        current_sigma = find_sigma(x_peak, y_peak, experimentData) / experimentData.gained_degree

        a_from_t_sum[t_sum] = current_sigma * (t_sum)**0.5
        a_from_step[step] = current_sigma * (t_sum) ** 0.5

        if current_sigma != 0:
            sigma[t_sum] = current_sigma

        if step <= 50 and prev_sigma == experimentData.F_max - experimentData.F_min and current_sigma != 0:
            flag = True

        if flag and \
                step - prev_step >= 1: # and \
                #prev_sigma + experimentData.delta_F/experimentData.gained_degree > 2 * current_sigma:# and \
                #experimentData.const * F * experimentData.F_degree * experimentData.t <= 3.14:
            prev_sigma = current_sigma
            prev_step = step
            experimentData.t *= experimentData.time_const
            print(step)

        if flag and prev_sigma < current_sigma:
            prev_sigma = current_sigma

        if (step) % 1 == 0:
            plt.plot([experimentData.F_min + i*experimentData.delta_F for i in range(experimentData.fields_number)], [each for each in experimentData.probability_distribution]) # distr each 50 steps

        if (step + 1) % 1 == 0:
            print(bayesians_learning.integrate_distribution(experimentData), x_peak, y_peak, step, current_sigma, prev_sigma, t_sum, experimentData.const * experimentData.F * experimentData.t*experimentData.F_degree, flag) # checking ~ 1

        if flag and current_sigma*experimentData.gained_degree <= 7*experimentData.delta_F:
            plt.plot([experimentData.F_min + i * experimentData.delta_F for i in range(experimentData.fields_number)],
                     [each for each in experimentData.probability_distribution])
            plt.show()
            plt.close()

            fig, ax = plt.subplots()
            ax.minorticks_on()

            expand(x_peak, y_peak, current_sigma, experimentData)
            plt.plot([experimentData.F_min + i * experimentData.delta_F for i in range(experimentData.fields_number)],
                     [each/10 for each in experimentData.probability_distribution])
            print(experimentData.probability_distribution)

        if y_peak >= 1.0 - epsilon or t_sum >= 100*10**(-3):
            break

    #plt.plot([experimentData.F_min + i*experimentData.delta_F for i in range(experimentData.fields_number)], experimentData.probability_distribution) # final distr
    plt.show()
    #fig.savefig('distr_' + '.png', dpi=500)
    plt.close()
    print(list(sigma.keys())[-1], list(sigma.values())[-1])

    try:
        plotter.plotting_sensitivity(a_from_step, r'$N$')
    except Exception:
        pass
    try:
        plotter.plotting_sensitivity(a_from_t_sum, r'$t_{sum}$')
    except Exception:
        pass

    print("final sensitivity: ", a_from_t_sum[t_sum]*10**(-9))

    x_peak, y_peak = find_peak(experimentData)

    plotter.plotting(sigma)

    #return x_peak, plotter.plotting(sigma)[1][1], list(sigma.keys())[-1], list(sigma.values())[-1]

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
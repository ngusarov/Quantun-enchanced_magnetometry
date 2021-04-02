import math
from multiplicating_t_phi_const import qubit
import matplotlib.pyplot as plt


# constants start ---------------------
F = 50  # field strength to be measured in Tesla
F_min, F_max = 1, 100 # 1 ... 10 Tesla
delta_F = 1 # accuracy of F defining
fields_number = int( ((F_max - F_min + delta_F)//delta_F) ) # amount of discrete F meanings
t = 4.44*10**(-7)*2**10 # time of interaction in seconds
flag = False
time_const = 2
mu = 10**5  # magnetic moment of the qubit
h = 6,62*10**(-34) # plank's constant
const = 10/3.14 #mu/h
# constants end -----------------------

# data of measures --------------------
probability_distribution = [ 1 / fields_number ]*fields_number # initial probability is equal for each field
# data of measures --------------------


def P_F_i(F_i, F_min, delta_F, distr):
    '''
    :param F_i: field in Tesla
    :param distr: current distribution of probabilities
    :return: probability for particular F_i in current distr
    '''
    return distr[int((F_i-F_min)//delta_F)]


def P_qubit_state_on_F_i(qubit_state, F_i, t):
    '''
    :param qubit_state: 1 or 0 after measuring
    :param F_i: field in Tesla
    :return: based on (1), (2) in ReadMe.md we return conditional probability
    '''
    #if flag:
    #    F_i /= time_const
    if qubit_state == 1: return ( math.cos(mu/math.pi*F_i*t) )**2
    else: return ( math.sin(mu/math.pi*F_i*t) )**2


def rect_integral_of_multiplication_probabilities(P_F_i, P_qubit_state_on_F_i, \
                                                  x_min, x_max, dx, distr, qubit_state):
    '''
    :param P_F_i: function
    :param P_qubit_state_on_F_i: function
    :param x_min: F_min
    :param x_max: F_max
    :param dx: delta_F
    :param distr: current distribution
    :param qubit_state: 0 or 1
    :return: integral of multiplication of functions from F_i
    '''
    integral = 0
    x = x_min
    for i in range( int((x_max - x_min)//dx) + 1): # TODO here is the BUG with amount of epochs
        integral += P_F_i(x, F_min, delta_F, distr)*P_qubit_state_on_F_i(qubit_state, x, t)*dx
        x += dx
    return integral


def reaccount_P_F_i(new_qubit_state, F_i, distr, t, F_min, delta_F):
    '''
    :param new_qubit_state: 1 or 0
    :param F_i: field in Tesla
    :param distr: current distribution
    :return: reaccounted probability of particular field based on Bayesian's Theorem
    '''
    return P_F_i(F_i, F_min, delta_F, distr) * P_qubit_state_on_F_i(new_qubit_state, F_i, t) / \
           rect_integral_of_multiplication_probabilities(P_F_i,
                                 P_qubit_state_on_F_i,
                                 F_min,
                                 F_max,
                                 delta_F, distr, new_qubit_state)


def renew_probalities(new_qubit_state, distr, F_min, delta_F, t):
    '''
    :param new_qubit_state: 0 or 1
    :param distr: current distribution of all fields
    :return: new distribution of all fields
    '''
    old_distr = distr # saving current meanings to reaccount all P(F_i) at once
    for i in range(fields_number):
        distr[i] = reaccount_P_F_i(new_qubit_state, F_min + delta_F*i, old_distr, t, F_min, delta_F)
    for i in range(fields_number): # normalizing values so as the sum would be = 1
        distr[i] = distr[i] * 1 / sum(distr)

    return distr


#--Finding sigma of the distrdibution-------------


def find_peak(distr):
    '''
    :param distr: distribution
    :return: x, y(x) # peak
    '''
    y_max = distr[0]
    x_max = F_min
    for i in range(1, fields_number):
        if distr[i] > y_max:
            x_max = F_min + delta_F*i
            y_max = distr[i]
    return x_max, y_max

def find_sigma(x_peak, y_peak, distr):
    '''
    :param x_peak:
    :param y_peak:
    :param distr:
    :return:
    '''
    y_sigma = y_peak / (2**0.5)
    epsilon = max([ abs( distr[i] - distr[i-1] )/2 for i in range(1, fields_number) ])
    x_sigma = []
    for i in range(1, fields_number):
        if abs(distr[i] - y_sigma) <= epsilon:
            x_sigma.append(F_min + delta_F*i)
    if len(x_sigma) > 0:
        return min([ abs(x_sigma[i] - x_peak) for i in range(len(x_sigma)) ])
    else:
        return 0

#--Finding sigma of the distrdibution-------------


if __name__ == '__main__':
    print(probability_distribution) # initial
    sigma = {}
    prev_sigma = F_max - F_min
    N = 8000
    t_sum = 0
    epsilon = 10**(-2)
    for step in range(N):

        probability_distribution = renew_probalities(qubit.randbin(t), probability_distribution, F_min, delta_F, t)
        t_sum += t

        x_peak, y_peak = find_peak(probability_distribution)
        current_sigma = find_sigma(x_peak, y_peak, probability_distribution)
        if current_sigma != 0:
            sigma[t_sum] = current_sigma

        if step >= 50 and prev_sigma+delta_F > 2*current_sigma:# and prev_sigma != 0:
            prev_sigma = current_sigma
            t *= time_const
            print(step)

        if (step+1)%10 == 0:
            plt.plot(range(F_min, F_max+delta_F, delta_F), probability_distribution) # distr each 50 steps
        if (step + 1) % 10 == 0:
            print(sum(probability_distribution), x_peak, y_peak, step, current_sigma, prev_sigma, t_sum) # checking ~ 1
        if y_peak >= 1.0 - epsilon:
            break


    plt.plot(range(1, fields_number+1), probability_distribution) # final distr
    plt.show()
    plt.close()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(sigma.keys(), [1 / (each)/80000 for each in sigma.keys()])
    plt.plot(sigma.keys(), [(1 / (each))**0.5/80 for each in sigma.keys()])
    plt.plot(sigma.keys(), sigma.values())
    plt.show()
    plt.close()
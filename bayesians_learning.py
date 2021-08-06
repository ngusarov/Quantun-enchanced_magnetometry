import math
import qubit
from sklearn import preprocessing
import numpy as np

def P_F_i(i, old_distr):
    '''
    :param F_i: field in Tesla
    :param data:
    :return: probability for particular F_i in current distr
    '''
    return old_distr[i]


def P_qubit_state_on_F_i(qubit_state, F_i, data):
    '''
    :param qubit_state: 1 or 0 after measuring
    :param F_i: field in Tesla
    :return: based on (1), (2) in ReadMe.md we return conditional probability
    '''
    '''
    if qubit_state == 1: return ( math.cos(data.const*F_i*data.F_degree*data.t/2) )**2
    else: return ( math.sin(data.const*F_i*data.F_degree*data.t/2) )**2
    '''
    p_1 = ( math.cos(data.const*F_i*data.F_degree*data.t/2) )**2
    p_0 = ( math.sin(data.const*F_i*data.F_degree*data.t/2) )**2
    return (sum(qubit_state)*p_1 + (data.num_of_repetitions - sum(qubit_state))*p_0)/data.num_of_repetitions



def rect_integral_of_multiplication_probabilities(P_F_i,
                                 P_qubit_state_on_F_i,
                                 qubit_state, old_distr, data):
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
    x_min = data.F_min
    x_max = data.F_max
    dx = data.delta_F

    x = x_min
    for i in range(data.fields_number):
        integral += P_F_i(i, old_distr)*P_qubit_state_on_F_i(qubit_state, x, data)*dx
        x += dx
    return integral


def integrate_distribution(data):
    integral = 0
    for i in range(data.fields_number):
        integral += data.probability_distribution[i] * data.delta_F # / data.gained_degree
    return integral


def reaccount_P_F_i(i, new_qubit_state, F_i, old_distr, data):
    '''
    :param new_qubit_state: 1 or 0
    :param F_i: field in Tesla
    :param old_distr: current distribution
    :return: reaccounted probability of particular field based on Bayesian's Theorem
    '''
    return P_F_i(i, old_distr) * P_qubit_state_on_F_i(new_qubit_state, F_i, data) / \
           rect_integral_of_multiplication_probabilities(P_F_i,
                                 P_qubit_state_on_F_i,
                                 new_qubit_state, old_distr, data)


def renew_probalities(data):
    '''
    :param new_qubit_state: 0 or 1
    :param distr: current distribution of all fields
    :return: new distribution of all fields
    '''

    #new_qubit_state = int(round(sum([qubit.randbin(data, data.F) for i in range(data.num_of_repetitions)])/data.num_of_repetitions))
    new_qubit_state = [qubit.randbin(data, data.F) for i in range(data.num_of_repetitions)]
    old_distr = data.probability_distribution # saving current meanings to reaccount all P(F_i) at once

    for i in range(data.fields_number-1, 0, -1):
        data.probability_distribution[i] = reaccount_P_F_i(i, new_qubit_state, data.F_min + data.delta_F*i, old_distr, data)
    data.probability_distribution = normalise(data)


def normalise(data):
    local_data = data
    s = integrate_distribution(local_data)
    for i in range(local_data.fields_number):
        local_data.probability_distribution[i] = local_data.probability_distribution[i] / s
    return local_data.probability_distribution

#arr = []
#print(normalise(arr), sum(normalise(arr)))
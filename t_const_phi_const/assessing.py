import math
from t_const_phi_const import qubit
import matplotlib.pyplot as plt


# constants start ---------------------
F = 20  # field strength to be measured in Tesla
F_min, F_max = 1, 100 # 1 ... 10 Tesla
delta_F = 1 # accuracy of F defining
fields_number = int( ((F_max - F_min)//delta_F) ) # amount of discrete F meanings
t = 6.5*10**(-7)  # time of interaction in seconds
mu = 10**5  # magnetic moment of the qubit
# constants end -----------------------

# data of measures --------------------
probability_distribution = [ 1 / fields_number ]*fields_number # initial probability is equal for each field
# data of measures --------------------


def P_F_i(F_i, distr):
    '''
    :param F_i: field in Tesla
    :param distr: current distribution of probabilities
    :return: probability for particular F_i in current distr
    '''
    return distr[int((F_i-F_min)//delta_F)]


def P_qubit_state_on_F_i(qubit_state, F_i):
    '''
    :param qubit_state: 1 or 0 after measuring
    :param F_i: field in Tesla
    :return: based on (1), (2) in ReadMe.md we return conditional probability
    '''
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
    for i in range( int((x_max - x_min)//dx) ):
        integral += P_F_i(x, distr)*P_qubit_state_on_F_i(qubit_state, x)*dx
        x += dx
    return integral


def reaccount_P_F_i(new_qubit_state, F_i, distr):
    '''
    :param new_qubit_state: 1 or 0
    :param F_i: field in Tesla
    :param distr: current distribution
    :return: reaccounted probability of particular field based on Bayesian's Theorem
    '''
    return P_F_i(F_i, distr) * P_qubit_state_on_F_i(new_qubit_state, F_i) / \
           rect_integral_of_multiplication_probabilities(P_F_i,
                                 P_qubit_state_on_F_i,
                                 F_min,
                                 F_max,
                                 delta_F, distr, new_qubit_state)


def renew_probalities(new_qubit_state, distr):
    '''
    :param new_qubit_state: 0 or 1
    :param distr: current distribution of all fields
    :return: new distribution of all fields
    '''
    old_distr = distr # saving current meanings to reaccount all P(F_i) at once
    for i in range(fields_number):
        distr[i] = reaccount_P_F_i(new_qubit_state, F_min + delta_F*i, old_distr)
    return distr

if __name__ == '__main__':
    print(probability_distribution) # initial
    for i in range(200):
        probability_distribution = renew_probalities(qubit.return_random_state(), probability_distribution)
        if i%50 == 0:
            plt.plot(range(1, fields_number + 1), probability_distribution) # distr each 50 steps
    print(sum(probability_distribution)) # checking ~ 1

    plt.plot(range(1, fields_number+1), probability_distribution) # final distr
    plt.show()
    plt.close()



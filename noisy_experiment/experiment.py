from dataclasses import dataclass
import matplotlib.pyplot as plt
import math

import bayesians_learning
import qubit
import plotter
#import ramsey_qubit

# constants start ---------------------
#F = 20  # field strength to be measured in Tesla


@dataclass
class ExperimentData:
    def __init__(self, F):
        self.F = F
        self.F_min = 70  # min field Tesla
        self.F_max = 200 # max field Tesla
        self.F_degree = 10**(-10)
        self.delta_F = 1  # accuracy of F defining
        self.fields_number = round( (self.F_max - self.F_min + self.delta_F) / self.delta_F ) # amount of discrete F meanings
        self.time_const = 2
        self.mu = 10 ** (5) * 927 * 10**(-26)  # magnetic moment of the qubit
        self.h = 6.62 * 10 ** (-34)  # plank's constant
        self.const = self.mu/self.h  # mu/h
        self.t = math.pi/4/(self.const*self.F_degree*(self.F_max - self.F_min)/2) * 10**(-1)  # time of interaction in seconds
        self.probability_distribution = [1 / self.fields_number] * self.fields_number
        
    
    
class OptimalAngles:
    def __init__(self):
        self.F_real = []
        self.F_model = []
        self.y_model = 0
        self.scaling = []
        self.theta_angles = []
        self.summ = 0
        self.N = 10
        
    
    def count(self, theta_sphere, phi_sphere, F):
        self.summ = 0
        data = perform(theta_sphere, phi_sphere, F)
        sigma = data[2]
        self.scaling.append(data[1])
        arr = [[i] + [j] + [k] for (i, j, k) in zip([math.sqrt((optimal_angles.F_real[0] - optimal_angles.F_model[0])**2)], 
                                                    optimal_angles.scaling, optimal_angles.theta_angles)]
        success_idx = (arr[0][1])**2/(arr[0][0]+1)/sigma * 10**(-0.7) * self.y_model
        self.summ += success_idx
        self.F_real = []
        self.F_model = []
        self.scaling = []
        self.theta_angles = []
        self.sigma = []
        self.y_model = []
            
        return {"success_idx": success_idx}
    
class OptimalData:
    data = []
        


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


def find_sigma(x_peak, y_peak, data):
    """
    :param x_peak:
    :param y_peak:
    :param data:
    :return: sigma
    """
    y_sigma = y_peak / (2**0.5)
    epsilon = max([ abs( data.probability_distribution[i] - data.probability_distribution[i-1] )/2 for i in range(1, data.fields_number) ])
    x_sigma = []
    for i in range(1, data.fields_number):
        if abs(data.probability_distribution[i] - y_sigma) <= epsilon:
            x_sigma.append(data.F_min + data.delta_F*i)
    if len(x_sigma) > 0:
        return min([ abs(x_sigma[i] - x_peak) for i in range(len(x_sigma)) ])
    else:
        return 0
# work with distribution -----------


def perform(theta_sphere, phi_sphere, F):
    experimentData = ExperimentData(F)
    F = experimentData.F
    sigma = {}
    N = 200
    t_sum = 0
    epsilon = 10 ** (-3)
    prev_sigma = experimentData.F_max - experimentData.F_min
    flag = False
    prev_step = 0
    #

    print(experimentData.probability_distribution) # initial
    print(experimentData.fields_number)
    for step in range(N):

        bayesians_learning.renew_probalities(qubit.randbin3(experimentData, F, theta_sphere, phi_sphere), experimentData, theta_sphere, phi_sphere)
        #bayesians_learning.renew_probalities(qubit.randbin(experimentData, F), experimentData)
        #bayesians_learning.renew_probalities(ramsey_qubit.output(experimentData.t), experimentData)
        t_sum += experimentData.t

        x_peak, y_peak = find_peak(experimentData)
        current_sigma = find_sigma(x_peak, y_peak, experimentData)
        if current_sigma != 0:
            sigma[t_sum] = current_sigma

        if step <= 50 and prev_sigma == experimentData.F_max - experimentData.F_min and current_sigma != 0:
            flag = True

        if flag and \
                step - prev_step >= 2 and \
                prev_sigma + experimentData.delta_F > 2 * current_sigma and\
                experimentData.const * F * experimentData.F_degree * experimentData.t <= math.pi:
            prev_sigma = current_sigma
            prev_step = step
            experimentData.t *= experimentData.time_const
            print(step)

        if flag and prev_sigma < current_sigma:
            prev_sigma = current_sigma

        if (step) % 3 == 0:
            plt.plot([experimentData.F_min + i*experimentData.delta_F for i in range(experimentData.fields_number)], experimentData.probability_distribution) # distr each 50 steps

        if (step + 1) % 1 == 0:
            print(sum(experimentData.probability_distribution), x_peak, y_peak, step, current_sigma, prev_sigma, t_sum, experimentData.const * F * experimentData.t*experimentData.F_degree, flag) # checking ~ 1

        if y_peak >= 1.0 - epsilon:
            break
    
    x_peak, y_peak = find_peak(experimentData)
    optimal_angles.F_model.append(x_peak)
    optimal_angles.y_model = y_peak
    optimal_angles.F_real.append(F)
    optimal_angles.theta_angles.append(theta_sphere)
    plt.plot([experimentData.F_min + i*experimentData.delta_F for i in range(experimentData.fields_number)], experimentData.probability_distribution) # final distr
    plt.show()
    #fig.savefig('distr_' + '.png', dpi=500)
    plt.close()
    print(list(sigma.keys())[-1], list(sigma.values())[-1])
    x_peak, y_peak = find_peak(experimentData)
    print(x_peak)
    #plotter.plotting(sigma)[1]
    x_peak, y_peak = find_peak(experimentData)
    
    
    return x_peak, plotter.plotting(sigma)[1][1], list(sigma.keys())[-1], list(sigma.values())[-1]

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
    optimal_angles = OptimalAngles()
    N = 5
    optimal_data = OptimalData()
    optimal_data_one_experiment = []
    theta_angles = [i/1000 for i in range(1900, 1971, 120)]
    phi_angles = [i/100 for i in range(0, 100, 20)]
    F_array = [i for i in range(150, 200, 100)]
    #for phi in phi_angles:
    for theta in theta_angles:
        summ = 0
        for F in F_array:
            sigma = 0
            for i in range(N):
                optimal_data_one_experiment = optimal_angles.count(theta, 0, F)
                summ += optimal_data_one_experiment["success_idx"]
                print(theta)
                print(theta)
                print(theta)
                print(theta)
                print(theta)
        optimal_data.data.append([summ/N/len(F_array), theta])
        print(optimal_data.data)
    data = optimal_data.data
    data.sort(key=lambda x: -x[0])
    print(*filter(lambda x: x[0] > 0.00001 , data))
    #print(data)
    experimentData = ExperimentData(0)
        
    
    #for i in range(200):
    #    average_20(50, experimentData.F_max)
    #    experimentData.F_max += 30
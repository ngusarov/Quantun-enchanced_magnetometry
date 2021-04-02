import math
import numpy as np


def randbin(data, F):
    p_0 = (math.sin(data.const * F * data.t)) ** 2
    return np.random.choice([0, 1], size=(1,1), p=[p_0, 1-p_0]).reshape(1)[0]

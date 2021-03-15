import random
import math

from multiplicating_t_phi_const import assessing


def return_random_state():
    r = random.randint(1, 10**5)
    p_0 = (math.sin(assessing.mu * assessing.F * assessing.t / math.pi)) ** 2 * 10 ** 5
    if r <= p_0: return 0
    else: return 1
# Qubit Magnetometry


# Magnetometry with t_{i+1} = 2 \cdot t_i, \phi -- const

Initially we have a range of fields, each field has the same probability to be the real one. The actual field to be measured lies in this range.

1) Qubit is prepared in |0>
2) H applied
3) *Interacrion with magnetic field*
4) H applied
5) Qubit is measured: it collapses to 0 with P_0 prob.
To 1 -- with P_1. 
6) Probability of each field is renewed according to the new outcome. If uncertainty ($\sigma$) of the distr has decreased twice, time is being doubled. 

We can get P_1 and P_0 from equations:

$$sin^2 \phi = P_0 = sin^2 (\mu * F * t / \pi)$$ (1)

$$cos^2 \phi = P_1 = cos^2 (\mu * F * t / \pi)$$ (2)

The whole thing is driven by experiment.py -- it calls all other functions in perform() func.

After each qubit measuring we in bayesians_learning.py in reaccount_P_F_i renew probability of each particular field according to 
BAYESIAN's THEOREM. renew_probabilities() renews all F_i per call.

Qubit behavior is simulated by qubit.py: return_random_state() returns state with probabilities (1) and (2) based on
constants in experiment.py

real_experiment.py -- experiments with pulses (outcome() func imitates qubit)
comp_plotter.py -- plots outcomes of real_experiment
ramsey_cubit.py -- draft version of real_experiment.py

problem_with_gates.py -- experiments to calibrate "qubit outcome" functions

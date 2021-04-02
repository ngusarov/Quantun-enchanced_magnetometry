# Magnetometry with t, \phi -- const

1) Qubit is prepared in |0>
2) H applied
3) *Interacrion with magnetic field*
4) H applied
5) Qubit is measured: it collapses to 0 with P_0 prob.
To 1 -- with P_1.

According to 1-5) we can get P_1 and P_0 from equations:

$$sin^2 \phi = P_0 = sin^2 (\mu * F * t / \pi)$$ (1)

$$cos^2 \phi = P_1 = cos^2 (\mu * F * t / \pi)$$ (2)

After each qubit measuring we in assessing.py in reaccount_P_F_i renew probability of particular field according to 
BAYESIAN's THEOREM. renew_probabilities() renews all F_i per call.

Qubit behavior is simulated by qubit.py: return_random_state() returns state with probabilities (1) and (2) based on
constants in assessing.py

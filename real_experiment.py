from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt

import numpy as np

import math


from scipy.optimize import curve_fit


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

from qiskit import pulse  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter  # This is Parameter Class for variable parameters.

'''detuning_Hz = data.F_degree*(data.const * data.F)
print("initial time: ", data.t_init*10**6, ' us')
print("detuning: ", detuning_Hz*10**(-6), ' MHz')
phi = detuning_Hz*data.t
print("initial angle: ", phi, ' radians')
print(data.const * data.F_min * data.t * data.F_degree)
print(data.const * data.F_max * data.t * data.F_degree)
print("probability to have 1: ", (math.cos(phi/2))**2)'''

#IBMQ.disable_account()
token ='1e59d98e02c0540ac85a62e0aeb450bb17e26bc72e68471cfc0c26a8b9fa3b3f6d20c55f26866c5a415203278e8fdae17b65aa9b798243da76b2d3b45add1193'
provider = IBMQ.enable_account(token)
#provider = IBMQ.get_provider(hub='ibm-q')
#provider = IBMQ.load_account()
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
#backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator
#                                    and x.configuration().pulses==True))
#print('Given qubit frequency ', backend.properties().frequency(0), ' GHz')

backend_config = backend.configuration()
backend_defaults = backend.defaults()

dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

# We will find the qubit frequency for the following qubit.
qubit = 0
# We will define memory slot channel 0.
mem_slot = 0

given_qubit_frequency = backend.properties().frequency(qubit)
print('Qubit declared frequency is: ', given_qubit_frequency/GHz, 'GHz')
#-------------------------------------------------------------------------------
# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = given_qubit_frequency        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14



#center_frequency_Hz = 4.97167 * GHz
# We will sweep 40 MHz around the estimated frequency
frequency_span_Hz = 20 * MHz
# in steps of 1 MHz.
frequency_step_Hz = 0.27 * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz,
                            frequency_max / GHz,
                            frequency_step_Hz / GHz)

print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz.")



# Drive pulse parameters (us = microseconds)
drive_sigma_sec = 0.075 * us  # This determines the actual width of the gaussian
drive_duration_sec = drive_sigma_sec * 8  # This is a truncating parameter, because gaussians don't have
# a natural finite length

#'''
drive_amp = 0.05

# Create the base schedule
# Start with drive pulse acting on the drive channel
freq = Parameter('freq')
with pulse.build(backend=backend, default_alignment='sequential', name='Frequency sweep') as sweep_sched:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(freq, drive_chan)
    # Drive pulse samples
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              sigma=drive_sigma,
                              amp=drive_amp,
                              name='freq_sweep_excitation_pulse'), drive_chan)
    # Define our measurement pulse
    pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

# Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_GHz * GHz
schedules = [sweep_sched.assign_parameters({freq: f}, inplace=False) for f in frequencies_Hz]

num_shots_per_frequency = 1024

job = backend.run(schedules,
                  meas_level=1,
                  meas_return='avg',
                  shots=num_shots_per_frequency)



job_monitor(job)

frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds



sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*scale_factor
    # Get the results for `qubit` from this experiment
    sweep_values.append(res[qubit])
#'''
#'''
fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}

plt.plot(frequencies_GHz, np.real(sweep_values), color='blue') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency, GHz", **font, fontsize=25)
plt.ylabel("Measured signal, a.u.", **font, fontsize=25)

fig.savefig('materials_for_article\\resonance_surve.pdf', dpi=500)

plt.show()
plt.close()
#'''


#'''
fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values),
                                 lambda x, A, q_freq, B, C: -(A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [0.005, 4.975, 0.001, 3.3] # initial parameters for curve_fit
                                )

plt.plot(frequencies_GHz, np.real(sweep_values), color='blue', label='measured signal')
plt.plot(frequencies_GHz, y_fit, color='black', label='fitting curve')
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])

plt.xlabel("Frequency, GHz")
plt.ylabel("Measured Signal, a.u.")
plt.title('Resonance curve', fontsize=15)
plt.legend(loc='best')
plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(given_qubit_frequency, 9)} Hz to {round(rough_qubit_frequency, 9)} Hz.")

print("Difference between given and found ", (rough_qubit_frequency - given_qubit_frequency)/MHz, " MHz")
#-------------------------------------------------------------------------------
#'''
#rough_qubit_frequency = 4971610574.809171
#'''
#print(f"We've updated our qubit frequency estimate from "
#      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 7)} GHz to {round(rough_qubit_frequency/GHz, 7)} GHz.")

#print("Difference between given and found ", (rough_qubit_frequency - given_qubit_frequency)/MHz, " MHz")


# This experiment uses these values from the previous experiment:
    # `qubit`,
    # `measure`, and
    # `rough_qubit_frequency`.
#'''
# Rabi experiment parameters
num_rabi_points = 75

# Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
drive_amp_min = 0
drive_amp_max = 0.4
drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

# Build the Rabi experiments:
#    A drive pulse at the qubit frequency, followed by a measurement,
#    where we vary the drive amplitude each time.

drive_amp = Parameter('drive_amp')
with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as rabi_sched:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(rough_qubit_frequency, drive_chan)
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              amp=drive_amp,
                              sigma=drive_sigma,
                              name='Rabi Pulse'), drive_chan)
    pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

rabi_schedules = [rabi_sched.assign_parameters({drive_amp: a}, inplace=False) for a in drive_amps]

num_shots_per_point = 1024

job = backend.run(rabi_schedules,
                  meas_level=1,
                  meas_return='avg',
                  shots=num_shots_per_point)

job_monitor(job)

rabi_results = job.result(timeout=120)



# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(values)

rabi_values = []
for i in range(num_rabi_points):
    # Get the results for `qubit` from the ith experiment
    rabi_values.append(rabi_results.get_memory(i)[qubit] * scale_factor)

rabi_values = np.real(baseline_remove(rabi_values))
#'''
'''
plt.xlabel("Drive amp [a.u.]")
plt.ylabel("Measured signal [a.u.]")
plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
plt.show()
'''

#'''

fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values,
                                 lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [3, 0.1, 0.3, 0])
fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}

plt.plot(drive_amps, rabi_values, '.', color='blue', label='measured signal')
plt.plot(drive_amps, y_fit, color='black', label='fitting curve')

drive_period = fit_params[2] # get period of rabi oscillation

plt.axvline(drive_period/2, color='blue', linestyle='--')
plt.axvline(drive_period, color='blue', linestyle='--')
plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')

plt.xlabel("Drive amp, a.u.", **font, fontsize=25)
plt.ylabel("Measured signal, a.u.", **font, fontsize=25)
plt.title('Rabi Experiment', **font, fontsize=25)
plt.legend(loc='best')
fig.savefig('materials_for_article\\rabi_osc.pdf', dpi=500)
plt.show()
plt.close()

pi_amp = abs(drive_period / 2)
#'''
#pi_amp = 0.147599566427070172380808799062
#pi_amp = 0.1516
print(f"Pi Amplitude = {format(pi_amp, '.30g')}")


#------------------------------------------------------------------------------------
#----------------------------------Discrimination------------------------------------
#------------------------------------------------------------------------------------

#'''
with pulse.build(backend) as pi_pulse:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              amp=pi_amp,
                              sigma=drive_sigma,
                              name='pi_pulse'), drive_chan)


# Create two schedules

# Ground state schedule
with pulse.build(backend=backend, default_alignment='sequential', name='ground state') as gnd_schedule:
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(rough_qubit_frequency, drive_chan)
    pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])


# Excited state schedule
with pulse.build(backend=backend, default_alignment='sequential', name='excited state') as exc_schedule:
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(rough_qubit_frequency, drive_chan)
    pulse.call(pi_pulse)
    pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

# Execution settings
num_shots = 1024

job = backend.run([gnd_schedule, exc_schedule],
                  meas_level=1,
                  meas_return='single',
                  shots=num_shots)

job_monitor(job)

gnd_exc_results = job.result(timeout=120)

gnd_results = gnd_exc_results.get_memory(0)[:, qubit]*scale_factor
exc_results = gnd_exc_results.get_memory(1)[:, qubit]*scale_factor

fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}

# Plot all the results
# All results from the gnd_schedule are plotted in blue
plt.scatter(np.real(gnd_results), np.imag(gnd_results),
                s=5, cmap='viridis', c='#069AF3', alpha=0.5, label='state |0>')
# All results from the exc_schedule are plotted in red
plt.scatter(np.real(exc_results), np.imag(exc_results),
                s=5, cmap='viridis', c='#FF6347', alpha=0.5, label='state |1>')

plt.axis('square')

# Plot a large dot for the average result of the 0 and 1 states.
mean_gnd = np.mean(gnd_results) # takes mean of both real and imaginary parts
mean_exc = np.mean(exc_results)

print("Ground mean: ", format(mean_gnd, '.30g'))
print("Excited mean: ", format(mean_exc, '.30g'))

plt.scatter(np.real(mean_gnd), np.imag(mean_gnd),
            s=200, cmap='viridis', c='#0000FF',alpha=1.0, label='state |0> mean')
plt.scatter(np.real(mean_exc), np.imag(mean_exc),
            s=200, cmap='viridis', c='#E50000',alpha=1.0, label='state |1> mean')

plt.ylabel('I, a.u.', **font, fontsize=25)
plt.xlabel('Q, a.u.', **font, fontsize=25)
plt.title("States discrimination", **font, fontsize=25)
plt.legend(loc='best')
plt.show()
fig.savefig('materials_for_article\\discrimination.pdf', dpi=500)
plt.close()

#'''
def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))



#mean_gnd = (2.96175354575296001513606825029+14.1656869588172789065083634341j)
#mean_exc = (-2.25588042700864033207608372322+14.4043790788198400321107328637j)

#'''
# Ramsey experiment parameters
time_max_sec = 1.8 * us
time_step_sec = 0.025 * us
delay_times_sec = np.arange(0.1 * us, time_max_sec, time_step_sec)

# Drive parameters
# The drive amplitude for pi/2 is simply half the amplitude of the pi pulse
drive_amp = pi_amp / 2

# x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
with pulse.build(backend) as x90_pulse:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              amp=drive_amp,
                              sigma=drive_sigma,
                              name='x90_pulse'), drive_chan)

detuning_MHz = 2
ramsey_frequency = round(rough_qubit_frequency + detuning_MHz * MHz, 6) # need ramsey freq in Hz

# create schedules for Ramsey experiment
ramsey_schedules = []
for delay in delay_times_sec:
    with pulse.build(backend=backend, default_alignment='sequential', name=f"Ramsey delay = {delay / ns} ns") as ramsey_schedule:
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(ramsey_frequency, drive_chan)
        pulse.call(x90_pulse)
        pulse.delay(get_closest_multiple_of_16(pulse.seconds_to_samples(delay)), drive_chan)
        pulse.call(x90_pulse)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
    ramsey_schedules.append(ramsey_schedule)

# Execution settings
num_shots = 256

job = backend.run(ramsey_schedules,
                  meas_level=1,
                  meas_return='single',
                  shots=num_shots)

job_monitor(job)

ramsey_results = job.result(timeout=120)

ramsey_values = []

for i in range(len(delay_times_sec)):
    iq_data = ramsey_results.get_memory(i)[:, qubit] * scale_factor
    ramsey_values.append(sum(map(classify, iq_data)) / num_shots)
#'''
'''
plt.scatter(delay_times_sec / us, np.real(ramsey_values), color='black')
plt.xlim(0, np.max(delay_times_sec / us))
plt.title("Ramsey Experiment", fontsize=15)
plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.show()
'''
#'''
fit_params, y_fit = fit_function(delay_times_sec/us, np.real(ramsey_values),
                                 lambda x, A, del_f_MHz, C, B: (
                                          A * np.cos(2*np.pi*del_f_MHz*x - C) + B
                                         ),
                                 [5, 1./0.4, 0, 0.25]
                                )

# Off-resonance component
_, del_f_MHz, _, _, = fit_params # freq is MHz since times in us

fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}

plt.plot(delay_times_sec/us, np.real(ramsey_values), '.', color='blue', label='measured signal')
plt.plot(delay_times_sec/us, y_fit, color='black', label='fitting curve')
plt.xlim(0, np.max(delay_times_sec/us))
plt.xlabel('Delay between X90 pulses, $\mu$s', **font, fontsize=25)
plt.ylabel('Measured Signal, a.u.', **font, fontsize=25)
plt.title('Ramsey Experiment', **font, fontsize=25)
plt.legend()
fig.savefig('materials_for_article\\ramsey_osc.pdf', dpi=500)
plt.show()
plt.close()

precise_qubit_freq = rough_qubit_frequency + (del_f_MHz - detuning_MHz) * MHz # get new freq in Hz
print(f"Our updated qubit frequency is now {round(precise_qubit_freq, 9)} Hz. "
      f"It used to be {round(rough_qubit_frequency, 9)} Hz")
#'''
#precise_qubit_freq = 4971592601.153201


#'''
import experiment
data = experiment.ExperimentData()
#'''

#'''
# Ramsey experiment parameters
detuning_max_MHz = data.const*data.F_max*data.F_degree/(2*math.pi)/MHz/30
delta_min_det_MHz = 0.0
#delta_max_det_MHz = -(detuning_max_MHz - delta_min_det_MHz)/(2**(2-1) + 3/4)*3/4
#delta_max_det_MHz = +(detuning_max_MHz - delta_min_det_MHz)/(2**(1-1) + 3/4)*3/4
delta_max_det_MHz = 0
detuning_s_MHz = np.arange(delta_min_det_MHz, detuning_max_MHz+delta_max_det_MHz, detuning_max_MHz/60)
print("max det: ", detuning_max_MHz)
time = data.t_init*(2**0)*30
print("time: ", time * 10**6)

# Drive parameters
# The drive amplitude for pi/2 is simply half the amplitude of the pi pulse
drive_amp = pi_amp / 2

# x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
with pulse.build(backend) as x90_pulse:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              amp=drive_amp,
                              sigma=drive_sigma,
                              name='x90_pulse'), drive_chan)

# create schedules for Ramsey experiment
ramsey_schedules = []
for detuning_MHz in detuning_s_MHz:
    ramsey_frequency = round(precise_qubit_freq + detuning_MHz * MHz, 6)  # need ramsey freq in Hz
    with pulse.build(backend=backend, default_alignment='sequential', name=f"det = {detuning_MHz} MHz") as ramsey_schedule:
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(ramsey_frequency, drive_chan)
        pulse.call(x90_pulse)
        pulse.delay(get_closest_multiple_of_16(pulse.seconds_to_samples(time)), drive_chan)
        pulse.call(x90_pulse)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
    ramsey_schedules.append(ramsey_schedule)

# Execution settings
num_shots = 256

job = backend.run(ramsey_schedules,
                  meas_level=1,
                  meas_return='single',
                  shots=num_shots)

job_monitor(job)

ramsey_results = job.result(timeout=120)

ramsey_values = []

for i in range(len(detuning_s_MHz)):
    iq_data = ramsey_results.get_memory(i)[:, qubit] * scale_factor
    ramsey_values.append(sum(map(classify, iq_data)) / num_shots)

fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}

plt.plot(detuning_s_MHz, np.real(ramsey_values), '.', color='blue', label='probability')
#plt.xlim(0, np.max(delay_times_sec / us))
#plt.title("Ramsey Experiment", fontsize=15)
plt.xlabel('Detuning, MHz', **font, fontsize=25)
plt.ylabel('$P_{|1>}$', **font, fontsize=25)
plt.legend(loc='best')
plt.show()
fig.savefig('materials_for_article\\impulse_probalts.pdf', dpi=500)
plt.close()
#'''
'''
fit_params, y_fit = fit_function(delay_times_sec/us, np.real(ramsey_values),
                                 lambda x, A, del_f_MHz, C, B: (
                                          A * np.cos(2*np.pi*del_f_MHz*x - C) + B
                                         ),
                                 [5, 1./0.4, 0, 0.25]
                                )

# Off-resonance component
_, del_f_MHz, _, _, = fit_params # freq is MHz since times in us

plt.scatter(detuning_s_MHz, np.real(ramsey_values), color='black')
plt.plot(detuning_s_MHz, y_fit, color='red', label=f"df = {del_f_MHz:.2f} MHz")
plt.xlim(0, np.max(delay_times_sec/us))
plt.xlabel('Detuning [MHz]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.title('Ramsey Experiment', fontsize=15)
plt.legend()
plt.show()
#'''

def output(data):
    print("We have F: ", data.F, " nT")

    N_delta = 3

    N = int(math.log(data.T_2 * 10 ** (-6) / data.t_init) / math.log(2))-N_delta
    multiplier = 30

    # Ramsey experiment parameters
    detuning_max_MHz = data.const * data.F_max * data.F_degree / (2 * math.pi) / MHz / multiplier
    detuning_min_MHz = data.const * data.F_min * data.F_degree / (2 * math.pi) / MHz / multiplier
    detuning_MHz = data.const * data.F * data.F_degree / (2 * math.pi) / MHz / multiplier

    #delta_min_det_MHz = +0.04
    #delta_max_det_MHz = -(detuning_max_MHz - delta_min_det_MHz)/(2**(3-1) + 2/3)*2/3

    #true_detuning_MHz = (detuning_min_MHz+delta_min_det_MHz) + (detuning_max_MHz+delta_max_det_MHz - (detuning_min_MHz+delta_min_det_MHz))*(detuning_MHz - detuning_min_MHz)/(detuning_max_MHz-detuning_min_MHz)

    times = [data.t_init*2**(i) for i in range(N)]

    # Drive parameters
    # The drive amplitude for pi/2 is simply half the amplitude of the pi pulse
    drive_amp = pi_amp / 2

    # x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
    with pulse.build(backend) as x90_pulse:
        drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
        drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
        drive_chan = pulse.drive_channel(qubit)
        pulse.play(pulse.Gaussian(duration=drive_duration,
                                  amp=drive_amp,
                                  sigma=drive_sigma,
                                  name='x90_pulse'), drive_chan)

    # create schedules for Ramsey experiment
    ramsey_schedules = []
    for i in range(len(times)):
        delta_min_det_MHz = +0.05
        if (i == 0): delta_max_det_MHz = -0.07
        elif (i<=4): delta_max_det_MHz = -(detuning_max_MHz - delta_min_det_MHz) / (2 ** (i - 1) + 3 / 4) * 3 / 4
        else: delta_max_det_MHz = +(detuning_max_MHz - delta_min_det_MHz) / (2 ** (i - 1) + 3 / 4) * 3 / 4

        true_detuning_MHz = (detuning_min_MHz + delta_min_det_MHz) + (
                    detuning_max_MHz + delta_max_det_MHz - (detuning_min_MHz + delta_min_det_MHz)) * (
                                        detuning_MHz - detuning_min_MHz) / (detuning_max_MHz - detuning_min_MHz)

        ramsey_frequency = round(precise_qubit_freq + true_detuning_MHz * MHz, 6)  # need ramsey freq in Hz

        with pulse.build(backend=backend, default_alignment='sequential',
                         name=f"det = {detuning_MHz} MHz") as ramsey_schedule:
            drive_chan = pulse.drive_channel(qubit)
            pulse.set_frequency(ramsey_frequency, drive_chan)
            pulse.call(x90_pulse)
            pulse.delay(get_closest_multiple_of_16(pulse.seconds_to_samples(times[i]*multiplier)), drive_chan)
            pulse.call(x90_pulse)
            pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
        ramsey_schedules.append(ramsey_schedule)

    # Execution settings
    num_shots = data.num_of_repetitions

    job = backend.run(ramsey_schedules,
                      meas_level=1,
                      meas_return='single',
                      shots=num_shots)

    job_monitor(job)

    ramsey_results = job.result(timeout=120)

    ramsey_values = {}

    for i in range(len(times)):
        iq_data = ramsey_results.get_memory(i)[:, qubit] * scale_factor
        ramsey_values[times[i]]=int(round(sum(map(classify, iq_data)) / num_shots))
    '''
    times = [data.t_init * 2 ** (i) for i in range(N, N+N_delta)]

    # create schedules for Ramsey experiment
    ramsey_schedules = []
    ramsey_frequency = round(precise_qubit_freq + detuning_MHz * MHz, 6)  # need ramsey freq in Hz
    for time in times:
        with pulse.build(backend=backend, default_alignment='sequential',
                         name=f"det = {detuning_MHz} MHz") as ramsey_schedule:
            drive_chan = pulse.drive_channel(qubit)
            pulse.set_frequency(ramsey_frequency, drive_chan)
            pulse.call(x90_pulse)
            pulse.delay(get_closest_multiple_of_16(pulse.seconds_to_samples(time * multiplier)), drive_chan)
            pulse.call(x90_pulse)
            pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
        ramsey_schedules.append(ramsey_schedule)

    # Execution settings
    num_shots = data.num_of_repetitions

    job = backend.run(ramsey_schedules,
                      meas_level=1,
                      meas_return='single',
                      shots=num_shots)

    job_monitor(job)

    ramsey_results = job.result(timeout=120)

    for i in range(len(times)):
        iq_data = ramsey_results.get_memory(i)[:, qubit] * scale_factor
        ramsey_values[times[i]] = int(round(sum(map(classify, iq_data)) / num_shots))
    #'''


    #print(ramsey_values)
    return ramsey_values

#print(output(data))
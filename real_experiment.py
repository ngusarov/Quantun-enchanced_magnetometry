from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

import numpy as np
import experiment
import math

data = experiment.ExperimentData()
'''detuning_Hz = data.F_degree*(data.const * data.F)
print("initial time: ", data.t_init*10**6, ' us')
print("detuning: ", detuning_Hz*10**(-6), ' MHz')
phi = detuning_Hz*data.t
print("initial angle: ", phi, ' radians')
print(data.const * data.F_min * data.t * data.F_degree)
print(data.const * data.F_max * data.t * data.F_degree)
print("probability to have 1: ", (math.cos(phi/2))**2)'''

#IBMQ.disable_account()
token ='d450d58f70726aa812595264cebdcc1b954e95cde187217ab4cbe3be5c27a3d330fb6a8fd34007762796f423d2fd7078952738e351a7828397cc184e48d86a6e'
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
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14



center_frequency_Hz = 4.97167 * GHz
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

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

from qiskit import pulse  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter  # This is Parameter Class for variable parameters.

# Drive pulse parameters (us = microseconds)
drive_sigma_sec = 0.075 * us  # This determines the actual width of the gaussian
drive_duration_sec = drive_sigma_sec * 8  # This is a truncating parameter, because gaussians don't have
# a natural finite length


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

from qiskit.tools.monitor import job_monitor

job_monitor(job)

frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds

import matplotlib.pyplot as plt

sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*scale_factor
    # Get the results for `qubit` from this experiment
    sweep_values.append(res[qubit])

plt.scatter(frequencies_GHz, np.real(sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured signal [a.u.]")
plt.show()

from scipy.optimize import curve_fit


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values),
                                 lambda x, A, q_freq, B, C: -(A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [0.005, 4.975, 0.001, 3.3] # initial parameters for curve_fit
                                )

plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
plt.plot(frequencies_GHz, y_fit, color='red')
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])

plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 9)} GHz to {round(rough_qubit_frequency/GHz, 9)} GHz.")

print("Difference between given and found ", (rough_qubit_frequency - given_qubit_frequency)/MHz, " MHz")
#-------------------------------------------------------------------------------

#rough_qubit_frequency = 4.97167 * GHz

#print(f"We've updated our qubit frequency estimate from "
#      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 7)} GHz to {round(rough_qubit_frequency/GHz, 7)} GHz.")

#print("Difference between given and found ", (rough_qubit_frequency - given_qubit_frequency)/MHz, " MHz")


# This experiment uses these values from the previous experiment:
    # `qubit`,
    # `measure`, and
    # `rough_qubit_frequency`.

# Rabi experiment parameters
num_rabi_points = 50

# Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
drive_amp_min = 0
drive_amp_max = 0.75
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

plt.xlabel("Drive amp [a.u.]")
plt.ylabel("Measured signal [a.u.]")
plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
plt.show()


fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values,
                                 lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [3, 0.1, 0.3, 0])

plt.scatter(drive_amps, rabi_values, color='black')
plt.plot(drive_amps, y_fit, color='red')

drive_period = fit_params[2] # get period of rabi oscillation

plt.axvline(drive_period/2, color='red', linestyle='--')
plt.axvline(drive_period, color='red', linestyle='--')
plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')

plt.xlabel("Drive amp [a.u.]", fontsize=15)
plt.ylabel("Measured signal [a.u.]", fontsize=15)
plt.show()

pi_amp = abs(drive_period / 2)

#pi_amp = 0.1409737344919153
#pi_amp = 0.14100094748281855
print(f"Pi Amplitude = {pi_amp}")


#------------------------------------------------------------------------------------
#----------------------------------Discrimination------------------------------------
#------------------------------------------------------------------------------------


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

plt.figure()

# Plot all the results
# All results from the gnd_schedule are plotted in blue
plt.scatter(np.real(gnd_results), np.imag(gnd_results),
                s=5, cmap='viridis', c='blue', alpha=0.5, label='state_0')
# All results from the exc_schedule are plotted in red
plt.scatter(np.real(exc_results), np.imag(exc_results),
                s=5, cmap='viridis', c='red', alpha=0.5, label='state_1')

plt.axis('square')

# Plot a large dot for the average result of the 0 and 1 states.
mean_gnd = np.mean(gnd_results) # takes mean of both real and imaginary parts
mean_exc = np.mean(exc_results)


#mean_gnd = (3.2613036429457596+14.105148448440321j)
#mean_exc = (-1.6603515402460798+16.63620406837248j)
print("Ground mean: ", mean_gnd)
print("Excited mean: ", mean_exc)

plt.scatter(np.real(mean_gnd), np.imag(mean_gnd),
            s=200, cmap='viridis', c='black',alpha=1.0, label='state_0_mean')
plt.scatter(np.real(mean_exc), np.imag(mean_exc),
            s=200, cmap='viridis', c='black',alpha=1.0, label='state_1_mean')

plt.ylabel('I [a.u.]', fontsize=15)
plt.xlabel('Q [a.u.]', fontsize=15)
plt.title("0-1 discrimination", fontsize=15)

plt.show()


def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))

'''
#mean_gnd = -1.66-1.5j # takes mean of both real and imaginary parts
#mean_exc = 0.44+7.45j
mean_gnd = (-17.6155951333376-9.4155731402752j)
mean_exc = (-13.75797167652864-15.203761096622081j)
'''

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

plt.scatter(delay_times_sec / us, np.real(ramsey_values), color='black')
plt.xlim(0, np.max(delay_times_sec / us))
plt.title("Ramsey Experiment", fontsize=15)
plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.show()

fit_params, y_fit = fit_function(delay_times_sec/us, np.real(ramsey_values),
                                 lambda x, A, del_f_MHz, C, B: (
                                          A * np.cos(2*np.pi*del_f_MHz*x - C) + B
                                         ),
                                 [5, 1./0.4, 0, 0.25]
                                )

# Off-resonance component
_, del_f_MHz, _, _, = fit_params # freq is MHz since times in us

plt.scatter(delay_times_sec/us, np.real(ramsey_values), color='black')
plt.plot(delay_times_sec/us, y_fit, color='red', label=f"df = {del_f_MHz:.2f} MHz")
plt.xlim(0, np.max(delay_times_sec/us))
plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.title('Ramsey Experiment', fontsize=15)
plt.legend()
plt.show()

precise_qubit_freq = rough_qubit_frequency + (del_f_MHz - detuning_MHz) * MHz # get new freq in Hz
print(f"Our updated qubit frequency is now {round(precise_qubit_freq/GHz, 6)} GHz. "
      f"It used to be {round(rough_qubit_frequency / GHz, 6)} GHz")


detuning_MHz = 2
ramsey_frequency = round(precise_qubit_freq + detuning_MHz * MHz, 6) # need ramsey freq in Hz
#fields_nT = np.linspace(data.F_min, data.F_max/2**5, 50)
data.t = 1*us
times = np.linspace(0, 2, 74)
#detunings_MHz = np.linspace(0, data.F_max*data.F_degree*data.const / MHz / 10, 74)
# for real detunings are not 0..2 but (0..2 + (2*0.288-1)/(2*2.212))*8.48
# create schedules for Ramsey experiment
ramsey_schedules = []
#phis = np.linspace(-2*math.pi, 2*math.pi, 74)
for time in times:
    #detuning_MHz = phi/(2*math.pi*MHz*data.t)
    #detuning_MHz = detuning_MHz - (2*0.1-1)/(2*2.0)
    time *= us
    print('detuning: ', detuning_MHz, ' MHz')
    print('phi: ', 2*math.pi*detuning_MHz*MHz*time, ' radians')
    ramsey_frequency = round(precise_qubit_freq + detuning_MHz*MHz, 9)
    print('rams freq: ', ramsey_frequency/GHz, ' GHz')
    with pulse.build(backend=backend, default_alignment='sequential', name=f"Ramsey detuning = {detuning_MHz} MHz") as ramsey_schedule:
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(ramsey_frequency, drive_chan)
        pulse.call(x90_pulse)
        pulse.delay(get_closest_multiple_of_16(pulse.seconds_to_samples(time)), drive_chan)
        print("Delay samples: ", pulse.seconds_to_samples(time), ' n')
        pulse.call(x90_pulse)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
    ramsey_schedules.append(ramsey_schedule)

# Execution settings
num_shots = 1024

job = backend.run(ramsey_schedules,
                  meas_level=1,
                  meas_return='single',
                  shots=num_shots)

job_monitor(job)

ramsey_results = job.result(timeout=120)

ramsey_values = []

for i in range(len(times)):
    iq_data = ramsey_results.get_memory(i)[:, qubit] * scale_factor
    ramsey_values.append(sum(map(classify, iq_data)) / num_shots)

plt.scatter(times, ramsey_values, color='black')
plt.title("Ramsey Experiment", fontsize=15)
plt.xlabel('Delay [$\mu$ s]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.show()


fit_params, y_fit = fit_function(times, np.real(ramsey_values),
                                 lambda x, A, del_f_MHz, C, B: (
                                          A * np.cos(2*np.pi*del_f_MHz*x - C*2*np.pi) + B
                                         ),
                                 [0.4, 1/6.28, 0.1, 0.4]
                                )

# Off-resonance component
_, del_f_MHz, C, _, = fit_params # freq is MHz since times in us

plt.scatter(times, np.real(ramsey_values), color='black')
plt.plot(times, y_fit, color='red', label=f"per = {del_f_MHz:.3f} MHz, ph = {C:.3f}")
plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
plt.ylabel('Measured Signal [a.u.]', fontsize=15)
plt.title('Ramsey Experiment', fontsize=15)
plt.legend()
plt.show()

#precise_qubit_freq = rough_qubit_frequency + (del_f_MHz - detuning_MHz) * MHz # get new freq in Hz
#print(f"Our updated qubit frequency is now {round(precise_qubit_freq/GHz, 6)} GHz. "
#      f"It used to be {round(rough_qubit_frequency / GHz, 6)} GHz")

'''
def output(t):
    # Ramsey experiment parameters
    t /= us
    time_max_us = 1.8
    time_step_us = 0.025
    times_us = np.full(60, t) #np.arange(t, t+time_step_us, time_step_us)
    print(times_us)
    # Convert to units of dt
    delay_times_dt = times_us * us / dt

    # create schedules for Ramsey experiment
    ramsey_schedules = []
    pulse.SetPhase(0, drive_chan)
    for delay in delay_times_dt:
        this_schedule = pulse.Schedule(name=f"Ramsey delay = {delay * dt / us} us")
        this_schedule |= Play(x90_pulse, drive_chan)
        this_schedule |= Play(x90_pulse, drive_chan) << int(this_schedule.duration + delay)
        this_schedule |= measure << int(this_schedule.duration)

        ramsey_schedules.append(this_schedule)

    # Execution settings
    num_shots = 1

    detuning_MHz = 10**(-6)*experiment.ExperimentData.F_degree*(experiment.ExperimentData.const * experiment.ExperimentData.F)#2
    ramsey_frequency = round(qubit_frequency + detuning_MHz * MHz, 6) # need ramsey freq in Hz
    ramsey_program = assemble(ramsey_schedules,
                                 backend=backend,
                                 meas_level=1,
                                 meas_return='avg',
                                 shots=num_shots,
                                 schedule_los=[{drive_chan: ramsey_frequency}]*len(ramsey_schedules)
                                )

    job = backend.run(ramsey_program)
    # print(job.job_id())
    job_monitor(job)

    ramsey_results = job.result(timeout=120)

    ramsey_values = []
    for i in range(len(times_us)):
        ramsey_values.append(ramsey_results.get_memory(i)[qubit]*scale_factor)
    print(ramsey_values)
    return ramsey_values # classify(ramsey_values[0])



count = 0
N = 1
# take time with phi ~  pi/4
# F = 5

val = output(data.t)

for i in range(len(val)):
    val[i] = classify(val[i])
print("final prob: ", 1 - sum(val) / len(val))

'''

'''for i in range(1, N):
    if output(data.t) == 0:
        count+=1
    print("prob: ", count/i)
'''
'''def output(t):
    for i in range(len(times_us)):
        if abs(t - times_us[i]) <= time_step_us/2:
            print('here')
            return classify(ramsey_values[i])'''


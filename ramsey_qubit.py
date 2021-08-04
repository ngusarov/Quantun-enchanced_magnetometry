import numpy as np
'''

from qiskit import IBMQ


provider = IBMQ.enable_account()
backend = provider.get_backend("ibmq_armonk")

backend_config = backend.configuration()
#assert backend_config.open_pulse, "Backend doesn't support Pulse"
backend_defaults = backend.defaults()

dt = backend_config.dt

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

# We will find the qubit frequency for the following qubit.
qubit = 0

# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = 4.97*GHz        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14

# We will sweep 40 MHz around the estimated frequency
frequency_span_Hz = 40 * MHz
# in steps of 1 MHz.
frequency_step_Hz = 1 * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz,
                            frequency_max / GHz,
                            frequency_step_Hz / GHz)

print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz.")


def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)


from qiskit import pulse            # This is where we access all of our Pulse features!
from qiskit.pulse import Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib


# Drive pulse parameters (us = microseconds)
drive_sigma_us = 0.075                     # This determines the actual width of the gaussian
drive_samples_us = drive_sigma_us*8        # This is a truncating parameter, because gaussians don't have
                                           # a natural finite length

drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)       # The width of the gaussian in units of dt
drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)   # The truncating parameter in units of dt
drive_amp = 0.05
# Drive pulse samples
drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                                 sigma=drive_sigma,
                                 amp=drive_amp,
                                 name='freq_sweep_excitation_pulse')

# Find out which group of qubits need to be acquired with this qubit
meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"

inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

### Collect the necessary channels
drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

# Create the base schedule
# Start with drive pulse acting on the drive channel
schedule = pulse.Schedule(name='Frequency sweep')
schedule += Play(drive_pulse, drive_chan)
# The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
schedule += measure << schedule.duration

# Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_GHz*GHz
schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]

from qiskit import assemble

num_shots_per_frequency = 1024
frequency_sweep_program = assemble(schedule,
                                   backend=backend,
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_frequency,
                                   schedule_los=schedule_frequencies)

job = backend.run(frequency_sweep_program)

# print(job.job_id())
from qiskit.tools.monitor import job_monitor
job_monitor(job)

frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds

sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*scale_factor
    # Get the results for `qubit` from this experiment
    sweep_values.append(res[qubit])

from scipy.optimize import curve_fit


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values),
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [-5, 4.975, 1, 5] # initial parameters for curve_fit
                                )
A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print("Rought cubit frequency: ", rough_qubit_frequency)

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
rabi_schedules = []
for drive_amp in drive_amps:
    rabi_pulse = pulse_lib.gaussian(duration=drive_samples, amp=drive_amp,
                                    sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
    this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
    this_schedule += Play(rabi_pulse, drive_chan)
    # Reuse the measure instruction from the frequency sweep experiment
    this_schedule += measure << this_schedule.duration
    rabi_schedules.append(this_schedule)

# Assemble the schedules into a Qobj
num_shots_per_point = 1024

rabi_experiment_program = assemble(rabi_schedules,
                                   backend=backend,
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_point,
                                   schedule_los=[{drive_chan: rough_qubit_frequency}]
                                                * num_rabi_points)

print(job.job_id())
job = backend.run(rabi_experiment_program)
job_monitor(job)

rabi_results = job.result(timeout=120)

# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(values)

rabi_values = []
for i in range(num_rabi_points):
    # Get the results for `qubit` from the ith experiment
    rabi_values.append(rabi_results.get_memory(i)[qubit]*scale_factor)

rabi_values = np.real(baseline_remove(rabi_values))

fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values,
                                 lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [3, 0.1, 0.5, 0])

drive_period = fit_params[2] # get period of rabi oscillation

pi_amp = abs(drive_period / 2)

pi_pulse = pulse_lib.gaussian(duration=drive_samples,
                              amp=pi_amp,
                              sigma=drive_sigma,
                              name='pi_pulse')

# Create two schedules

# Ground state schedule
gnd_schedule = pulse.Schedule(name="ground state")
gnd_schedule += measure

# Excited state schedule
exc_schedule = pulse.Schedule(name="excited state")
exc_schedule += Play(pi_pulse, drive_chan)  # We found this in Part 2A above
exc_schedule += measure << exc_schedule.duration

# Execution settings
num_shots = 1024

gnd_exc_program = assemble([gnd_schedule, exc_schedule],
                           backend=backend,
                           meas_level=1,
                           meas_return='single',
                           shots=num_shots,
                           schedule_los=[{drive_chan: rough_qubit_frequency}] * 2)

# print(job.job_id())
job = backend.run(gnd_exc_program)
job_monitor(job)

gnd_exc_results = job.result(timeout=120)

gnd_results = gnd_exc_results.get_memory(0)[:, qubit]*scale_factor
exc_results = gnd_exc_results.get_memory(1)[:, qubit]*scale_factor

mean_gnd = np.mean(gnd_results) # takes mean of both real and imaginary parts
mean_exc = np.mean(exc_results)
print(mean_gnd)
print(mean_exc)

import math

def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))

# T1 experiment parameters
time_max_us = 450
time_step_us = 6
times_us = np.arange(1, time_max_us, time_step_us)
# Convert to units of dt
delay_times_dt = times_us * us / dt
# We will use the same `pi_pulse` and qubit frequency that we calibrated and used before

# Create schedules for the experiment
t1_schedules = []
for delay in delay_times_dt:
    this_schedule = pulse.Schedule(name=f"T1 delay = {delay * dt/us} us")
    this_schedule += Play(pi_pulse, drive_chan)
    this_schedule |= measure << int(delay)
    t1_schedules.append(this_schedule)

sched_idx = 0

# Execution settings
num_shots = 256

t1_experiment = assemble(t1_schedules,
                         backend=backend,
                         meas_level=1,
                         meas_return='avg',
                         shots=num_shots,
                         schedule_los=[{drive_chan: rough_qubit_frequency}] * len(t1_schedules))

job = backend.run(t1_experiment)
# print(job.job_id())
job_monitor(job)

t1_results = job.result(timeout=120)

t1_values = []
for i in range(len(times_us)):
    t1_values.append(t1_results.get_memory(i)[qubit]*scale_factor)
t1_values = np.real(t1_values)

# Fit the data
fit_params, y_fit = fit_function(times_us, t1_values,
            lambda x, A, C, T1: (A * np.exp(-x / T1) + C),
            [-3, 3, 100]
            )

_, _, T1 = fit_params
'''
# Ramsey experiment parameters
time_max_us = 1.8
time_step_us = 0.025
times_us = np.arange(0.1, time_max_us, time_step_us) # here the only time t_of_interation`
# Convert to units of dt
delay_times_dt = times_us * us / dt

# Drive parameters
# The drive amplitude for pi/2 is simply half the amplitude of the pi pulse
drive_amp = pi_amp / 2
# x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
x90_pulse = pulse_lib.gaussian(duration=drive_samples,
                               amp=drive_amp,
                               sigma=drive_sigma,
                               name='x90_pulse')

# create schedules for Ramsey experiment
ramsey_schedules = []

for delay in delay_times_dt:
    this_schedule = pulse.Schedule(name=f"Ramsey delay = {delay * dt / us} us")
    this_schedule |= Play(x90_pulse, drive_chan)
    this_schedule |= Play(x90_pulse, drive_chan) << int(this_schedule.duration + delay)
    this_schedule |= measure << int(this_schedule.duration)

    ramsey_schedules.append(this_schedule)

# Execution settings
num_shots = 1

import experiment
detuning_MHz = 10**(-6)*experiment.ExperimentData.F_degree*(experiment.ExperimentData.const * experiment.ExperimentData.F)#2
ramsey_frequency = round(rough_qubit_frequency + detuning_MHz * MHz, 6) # need ramsey freq in Hz
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

fit_params, y_fit = fit_function(times_us, np.real(ramsey_values),
                                 lambda x, A, del_f_MHz, C, B: (
                                          A * np.cos(2*np.pi*del_f_MHz*x - C) + B
                                         ),
                                 [5, 1./0.4, 0, 0.25]
                                )

# Off-resonance component
_, del_f_MHz, _, _, = fit_params # freq is MHz since times in us

precise_qubit_freq = rough_qubit_frequency + (del_f_MHz - detuning_MHz) * MHz # get new freq in Hz

print()
print()
print(experiment.ExperimentData.t*10**6)

print()

#-----------------------------------------------------
mean_gnd = (-17.93922008219648-9.25825521942528j)
mean_exc = (-17.617234688081922-9.536411874099201j)

'''
two values above are constants for a single qubit. They are prepared beforehand.
Then we get new time of interaction -> get new detuning variable. Using it we perform a single
measurement for that particular time of interaction: above like we got series for ramsey_exp
using it we get new electro-magnetic wave or just a comlex number, though using the classificator
we recognize it as |0> or |1>
'''

time_max_us = 1.8
time_step_us = 0.025
times_us = np.arange(0.1, time_max_us, time_step_us)
ramsey_values = [(-6.45491104677888-18.87569407115264j), (-6.78123125342208-18.86929725423616j), (-6.68924244918272-19.00267746361344j), (-6.78794952179712-18.75218423349248j), (-6.45172606009344-18.82045676519424j), (-6.39810003795968-18.8180770848768j), (-6.87344151691264-18.96514079162368j), (-6.349471612928-18.95663944073216j), (-6.79788498911232-18.76521140617216j), (-6.2908620865536-18.84539307687936j), (-6.51692030820352-19.09960816459776j), (-6.5232735043584-19.0565752766464j), (-6.5889730822144-18.92394937090048j), (-6.5848358207488-18.89148076032j), (-6.8046911700992-18.71561661349888j), (-6.21682289016832-18.95902717411328j), (-6.4058128596992-18.84977528569856j), (-6.53220703633408-19.04591436251136j), (-6.81624127668224-18.97284891574272j), (-6.7587719299071995-18.83858958024704j), (-6.6832073490432-18.71290675757056j), (-6.8959659360256-18.86253939163136j), (-6.8463691300864-18.73813163737088j), (-6.4353172717568-18.85306093568j), (-6.84037093982208-18.75181513474048j), (-6.71860123500544-18.90364491300864j), (-6.3409957634048-18.89888018366464j), (-6.4701400612864-19.01900236587008j), (-6.56952292016128-18.75743483101184j), (-6.92680648556544-18.818600534016j), (-6.84625705828352-18.7958372073472j), (-6.46090722377728-18.96315705360384j), (-6.34525918953472-18.97873302093824j), (-6.83992131043328-18.804977434624j), (-6.64976565993472-18.815983288319998j), (-6.89388690341888-18.75495985610752j), (-6.71836836724736-18.81849181765632j), (-6.61002446176256-18.84537965510656j), (-6.52606590418944-18.83091501056j), (-6.89085828038656-18.765215432704j), (-6.7106481635328-18.868126875648j), (-6.59166683201536-18.9497044107264j), (-6.68281610436608-18.76156739485696j), (-6.59609937248256-18.57842596282368j), (-6.82401986510848-18.93949983686656j), (-6.83791408431104-18.84781570686976j), (-6.84703283675136-18.87214401224704j), (-6.76177303830528-18.73196030623744j), (-6.8129656930304-18.8634695204864j), (-6.7749116116992-18.7547867152384j), (-6.81164499058688-18.75103264538624j), (-7.1198075518976-18.75809249787904j), (-6.87195505557504-18.87690203070464j), (-6.76094894145536-18.76900976787456j), (-6.89204946272256-18.77970557861888j), (-6.69290390880256-18.70739980419072j), (-6.45756654452736-18.9697833828352j), (-6.33589146320896-18.8075342823424j), (-6.91532080349184-19.04636667625472j), (-6.9660242345984-18.8390217613312j), (-6.45485333315584-18.99372648333312j), (-6.9389290307584-19.00093129097216j), (-7.1268304945152-18.6368227540992j), (-6.7540843757568-18.56635442036736j), (-7.00426353639424-18.78314557898752j), (-6.77350299664384-18.73336288149504j), (-6.63964228780032-18.73591033397248j), (-6.9116325003264-18.8114333073408j)]


import math

def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))


def output(t):
    for i in range(len(times_us)):
        print(times_us[i])
        if abs(t - times_us[i]) <= time_step_us/2:
            return classify(ramsey_values[i])
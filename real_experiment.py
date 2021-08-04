from qiskit import IBMQ
import numpy as np
import experiment
from qiskit.providers.ibmq import least_busy
import math
data = experiment.ExperimentData()
print("time: ", experiment.ExperimentData.t*10**6 )
print("detuning: ", 10**(-6)*experiment.ExperimentData.F_degree*(experiment.ExperimentData.const * experiment.ExperimentData.F))
phi = data.const * data.F * data.t * data.F_degree
print("angle: ", phi)
print(data.const * data.F_min * data.t * data.F_degree)
print(data.const * data.F_max * data.t * data.F_degree)
print("probability: ", (math.cos(phi))**2)

IBMQ.disable_account()
tken
IBMQ.enable_account(tken)
provider = IBMQ.get_provider(hub='ibm-q')
#provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
#backend = least_busy(provider.backends(filters=lambda x: not x.configuration().simulator
#                                    and x.configuration().pulses==True))
print(backend.properties().frequency(0))

backend_config = backend.configuration()
backend_defaults = backend.defaults()

dt = backend_config.dt

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

# We will find the qubit frequency for the following qubit.
qubit = 0

qubit_frequency = backend.properties().frequency(qubit)
rough_qubit_frequency = round(qubit_frequency, 3)

print(f"Qubit {qubit} has an estimated frequency of {qubit_frequency / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14


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


from qiskit import assemble


from qiskit.tools.monitor import job_monitor



from scipy.optimize import curve_fit


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit



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
'''
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
                                   schedule_los=[{drive_chan: qubit_frequency}]
                                                * num_rabi_points)


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
                                 lambda x, A, B, drive_period, phi: (A*np.sin(2*np.pi*x/drive_period - phi) + B),
                                 [3, 0.1, 0.5, 0])

drive_period = fit_params[2] # get period of rabi oscillation

pi_amp = abs(drive_period / 2)'''
pi_amp = 0.1409737344919153
print(pi_amp)
'''
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
'''
#mean_gnd = -1.66-1.5j # takes mean of both real and imaginary parts
#mean_exc = 0.44+7.45j
mean_gnd = (-17.6155951333376-9.4155731402752j)
mean_exc = (-13.75797167652864-15.203761096622081j)
print(mean_gnd)
print(mean_exc)

def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))

# Drive parameters
# The drive amplitude for pi/2 is simply half the amplitude of the pi pulse
drive_amp = pi_amp / 2
# x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
x90_pulse = pulse_lib.gaussian(duration=drive_samples,
                                   amp=drive_amp,
                                   sigma=drive_sigma,
                                   name='x90_pulse')

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
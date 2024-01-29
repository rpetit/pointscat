import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from pointscat.forward_problem import PointScatteringProblem, compute_far_field
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure


np.random.seed(0)  # fix random seed

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['+Computer Modern'],
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# setting problem
wave_number = 1
box_size = 10  # locations should belong to (-box_size/2,box_size/2)
amplitudes = 0.5 * np.array([1, 2, 1, 0.5, 3, 1, 0.6, 2, 0.8])
locations = np.array([[-4.3, -4.7], [-4.0, 4.5], [4.2, 3.6], [0.1, 0.1], [2.5, 2.1], [-1.2, 3.4],
                      [-1/0.4, 1/0.4], [1/0.4, -1/0.4], [-1, -2]])

point_scat = PointScatteringProblem(locations, amplitudes, wave_number)
measure = DiscreteMeasure(locations, amplitudes)

# far field computation
num_frequencies = 100
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0] + 1j * k[1]) - np.arccos(np.linalg.norm(k) / (2 * wave_number))
                            for k in frequencies])
observation_directions = np.array([np.angle(k[0] + 1j * k[1]) + np.arccos(np.linalg.norm(k) / (2 * wave_number))
                                   for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)

# observations
obs = np.concatenate([np.real(far_field), -np.imag(far_field)])
born_obs = np.concatenate([np.real(far_field_born), -np.imag(far_field_born)])

print("relative L2 error between true far field and Born approximation:")
print(np.linalg.norm(obs - born_obs) / np.linalg.norm(born_obs))

std_noise = 0.1
noise = np.random.normal(scale=std_noise, size=obs.shape)
noisy_obs = obs + noise

print("relative noise level: {}".format(np.linalg.norm(noise) / np.linalg.norm(obs)))

# parameters
reg_param_nonlin = 5.0
tol_amplitudes = 0.01

x_tab = np.linspace(-box_size/2, box_size/2, 4)
init_locations = np.array([[x_1, x_2] for (x_1, x_2) in product(x_tab, x_tab)])
init_amplitudes = 0.3 * np.ones(len(init_locations))
estimated_measure = DiscreteMeasure(init_locations, init_amplitudes)

# display initialization
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes, label='unknown')

stemlines.set_color('black')
markerline.set_color('black')
baseline.set_linestyle('none')

markerline, stemlines, baseline = ax.stem(estimated_measure.locations[:, 0],
                                          estimated_measure.locations[:, 1],
                                          estimated_measure.amplitudes,
                                          label='estimated')

stemlines.set_color('blue')
markerline.set_color('blue')
baseline.set_linestyle('none')

ax.set_xlim(-1.1 * box_size / 2, 1.1 * box_size / 2)
ax.set_ylim(-1.1 * box_size / 2, 1.1 * box_size / 2)
ax.set_zlim(0, 1.6)

plt.show()

print("nonlinear sliding...")
state = estimated_measure.perform_nonlinear_sliding(incident_angles, observation_directions,
                                                    noisy_obs, wave_number,
                                                    box_size, reg_param=reg_param_nonlin,
                                                    tol_amplitudes=tol_amplitudes)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)

print("number of iterations:")
print(state.iter_num)

print("infinity norm of the gradient of the objective:")
print(np.max(np.abs(state.grad)))

print("relative l2 error on the observations:")
estimated_far_field = compute_far_field(estimated_measure.locations,
                                        estimated_measure.amplitudes,
                                        wave_number,
                                        incident_angles,
                                        observation_directions)
estimated_obs = np.concatenate([np.real(estimated_far_field), -np.imag(estimated_far_field)])
print(np.linalg.norm(obs - estimated_obs) / np.linalg.norm(obs))

# display output of nonlinear step
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(estimated_measure.locations[:, 0],
                                          estimated_measure.locations[:, 1],
                                          estimated_measure.amplitudes)

stemlines.set_color('red')
markerline.set_color('red')
baseline.set_linestyle('none')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes, label='unknown')

stemlines.set_color('black')
stemlines.set_alpha(0.7)
markerline.set_color('black')
markerline.set_alpha(0.7)
baseline.set_linestyle('none')

ax.set_xlim(-1.1 * box_size / 2, 1.1 * box_size / 2)
ax.set_ylim(-1.1 * box_size / 2, 1.1 * box_size / 2)
ax.set_zlim(0, 1.6)

plt.show()

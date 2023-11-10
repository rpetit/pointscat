import numpy as np
import matplotlib.pyplot as plt

from pointscat.forward_problem import PointScatteringProblem
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure, solve_blasso


np.random.seed(0)  # fix random seed

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['+Computer Modern'],
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# setting problem
amplitudes = np.array([1.2, 2.3, 1.9])
locations = np.array([[-0.25, 0.925], [-1.5, -1.0], [1.5, -0.5]])
wave_number = 1

point_scat = PointScatteringProblem(locations, amplitudes, wave_number)
measure = DiscreteMeasure(locations, amplitudes)

box_size = 5  # locations should belong to (-box_size/2,box_size/2)

# display unknown measure
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes, label='unknown')

stemlines.set_color('black')
markerline.set_color('black')
baseline.set_linestyle('none')

ax.set_xlim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_ylim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_zlim(0, 2.2)

plt.show()
# plt.savefig('meas_1.png', bbox_inches='tight', transparent=True, dpi=300)

# far field computation
num_frequencies = 30
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.linalg.norm(k)/(2*wave_number))
                            for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.linalg.norm(k)/(2*wave_number))
                                   for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)

# observations
obs = np.concatenate([np.real(far_field), -np.imag(far_field)])
born_obs = np.concatenate([np.real(far_field_born), -np.imag(far_field_born)])

print("L2 norm of observations:")
print(np.linalg.norm(obs))

print("relative L2 error between true far field and Born approximation:")
print(np.linalg.norm(obs - born_obs) / np.linalg.norm(obs))

print("absolute L2 error between true far field and Born approximation:")
print(np.linalg.norm(obs - born_obs))

std_noise = 0.25
noise = np.random.normal(scale=std_noise, size=obs.shape)
noisy_obs = obs + noise

print("relative L2 noise level")
print(np.linalg.norm(noise) / np.linalg.norm(obs))

# parameters
reg_param = 3.0
tol_locations = 0.05
tol_amplitudes = 0.01

# computation of the BLASSO estimator under Born approx
num_iter = 10

print("computation of the BLASSO estimator under Born approx...")
estimated_measure = solve_blasso(frequencies, noisy_obs, reg_param, num_iter, box_size,
                                 tol_locations=tol_locations, tol_amplitudes=tol_amplitudes)

# display output of linear step
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(estimated_measure.locations[:, 0],
                                          estimated_measure.locations[:, 1],
                                          estimated_measure.amplitudes,
                                          label='estimated')

stemlines.set_color('blue')
stemlines.set_alpha(0.5)
markerline.set_color('blue')
markerline.set_alpha(0.5)
baseline.set_linestyle('none')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes, label='unknown')

stemlines.set_color('black')
markerline.set_color('black')
baseline.set_linestyle('none')

ax.set_xlim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_ylim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_zlim(0, 2.2)

# plt.show()
plt.savefig('lin_est_1.png', bbox_inches='tight', transparent=True, dpi=300)

# nonlinear sliding step
print("nonlinear sliding...")
state = estimated_measure.perform_nonlinear_sliding(incident_angles, observation_directions,
                                                    noisy_obs, wave_number,
                                                    box_size, reg_param=reg_param,
                                                    tol_locations=tol_locations,
                                                    tol_amplitudes=tol_amplitudes)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)

# display output of nonlinear step
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(estimated_measure.locations[:, 0],
                                          estimated_measure.locations[:, 1],
                                          estimated_measure.amplitudes)

stemlines.set_color('red')
stemlines.set_alpha(0.5)
markerline.set_color('red')
markerline.set_alpha(0.5)
baseline.set_linestyle('none')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes)

stemlines.set_color('black')
markerline.set_color('black')
baseline.set_linestyle('none')

ax.set_xlim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_ylim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_zlim(0, 2.2)

# plt.show()
plt.savefig('nonlin_est_1.png', bbox_inches='tight', transparent=True, dpi=300)

import numpy as np
import matplotlib.pyplot as plt

from pointscat.forward_problem import angle_to_vec, PointScatteringProblem
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure, solve_blasso


np.random.seed(0)

# setting problem
amplitudes = 1 * np.array([1, 2, 1, 0.5, 3, 1, 0.6, 2])
locations = 0.4 * np.array([[-4.3, -4.7], [-4.0, 4.5], [4.2, 3.6], [0, 0], [2.5, 2.1], [-1.2, 3.4],
                            [-1/0.4, 1/0.4], [1/0.4, -1/0.4]])
wave_number = 1
point_scat = PointScatteringProblem(locations, amplitudes, wave_number)

# far field computation
box_size = 0.4 * 10  # locations should belong to (-box_size/2,box_size/2)
num_frequencies = 25
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.linalg.norm(k)/(2*wave_number))
                            for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.linalg.norm(k)/(2*wave_number))
                                   for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)

measure = DiscreteMeasure(locations, amplitudes)

# check relation between frequencies and difference of incident angles and observation directions
diff = np.array([wave_number * (angle_to_vec(observation_directions[i]) - angle_to_vec(incident_angles[i]))
                 for i in range(len(frequencies))])
diff = diff - frequencies
print(np.max(np.abs(diff)))

# check difference between Fourier transform and far field pattern
ft = measure.compute_fourier_transform(frequencies)
born_obs = np.concatenate([np.real(far_field_born), -np.imag(far_field_born)])
diff = ft - born_obs
print("difference between Fourier transform and far field pattern: {}".format(np.max(np.abs(diff))))

# check Born approximation error
print("relative L2 error between true far field and Born approximation:")
print(np.linalg.norm(far_field - far_field_born) / np.linalg.norm(far_field))

print("true amplitudes:")
print(measure.amplitudes)

print("true locations:")
print(measure.locations)

# computation of the BLASSO estimator under Born approx
reg_param = 0.1
num_iter = 8

print("computation of the BLASSO estimator under Born approx...")
obs = np.concatenate([np.real(far_field), -np.imag(far_field)])
estimated_measure = solve_blasso(frequencies, obs, reg_param, num_iter, box_size)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)

print("nonlinear sliding...")
estimated_measure.perform_nonlinear_sliding(incident_angles, observation_directions, obs, wave_number, box_size)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)

fig = plt.figure(figsize=plt.figaspect(3.))

ax = fig.add_subplot(1, 1, 1, projection='3d')

markerline, stemlines, baseline = ax.stem(locations[:, 0], locations[:, 1], amplitudes)

stemlines.set_color('black')
markerline.set_color('black')
baseline.set_linestyle('none')

markerline, stemlines, baseline = ax.stem(estimated_measure.locations[:, 0],
                                          estimated_measure.locations[:, 1],
                                          estimated_measure.amplitudes)

stemlines.set_color('red')
markerline.set_color('red')
baseline.set_linestyle('none')

ax.set_xlim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_ylim(-1.1*box_size/2, 1.1*box_size/2)
ax.set_title('Unknown and \nestimated measures')

fig.tight_layout()
plt.show()


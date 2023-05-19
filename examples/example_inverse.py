import numpy as np

from pointscat.forward_problem import angle_to_vec, PointScatteringProblem
from pointscat.inverse_problem import frequency_grid, DiscreteMeasure, solve_blasso


np.random.seed(0)

# setting problem
amplitudes = np.array([1, 1])
locations = np.array([[-1.6, -0.2], [2.2, 1.6]])
wave_number = 1
point_scat = PointScatteringProblem(wave_number, amplitudes, locations)

# far field computation
box_size = 50  # locations should belong to (-# box_size/2,box_size/2)
cutoff_frequency = int(box_size * wave_number / (np.pi * np.sqrt(2)))
frequencies = frequency_grid(cutoff_frequency)
print("cutoff frequency: {}".format(cutoff_frequency))

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.pi*np.linalg.norm(k)/(box_size*wave_number)) for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.pi*np.linalg.norm(k)/(box_size*wave_number)) for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)

measure = DiscreteMeasure(locations/box_size + 0.5 * np.ones(2)[np.newaxis, :], amplitudes)

# check relation between frequencies and difference of incident angles and observation directions
diff = np.array([wave_number * (angle_to_vec(observation_directions[i]) - angle_to_vec(incident_angles[i]))
                 for i in range(len(frequencies))])
diff = diff - frequencies * 2*np.pi/box_size
# print(np.max(np.abs(diff)))

# check difference between Fourier transform and far field pattern
ft = measure.compute_fourier_transform(frequencies)
born_obs = np.concatenate([np.real(far_field_born), -np.imag(far_field_born)]) * np.concatenate([(-1)**np.sum(frequencies, axis=1), (-1)**np.sum(frequencies, axis=1)])
diff = ft - born_obs
print("difference between Fourier transform and far field pattern: {}".format(np.max(np.abs(diff))))

# check Born approximation error
# print(np.linalg.norm(far_field - far_field_born) / np.linalg.norm(far_field))

# computation of BLASSO estimator
reg_param = 50.0
num_iter = 2

obs = np.concatenate([np.real(far_field), -np.imag(far_field)]) * np.concatenate([(-1)**np.sum(frequencies, axis=1), (-1)**np.sum(frequencies, axis=1)])
estimated_measure = solve_blasso(frequencies, obs, reg_param, num_iter)

print("true amplitudes:")
print(measure.amplitudes)

print("true locations:")
print(measure.locations)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)


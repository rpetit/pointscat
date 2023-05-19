import numpy as np

from pointscat.forward_problem import angle_to_vec, PointScatteringProblem
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure, solve_blasso


np.random.seed(0)

# setting problem
amplitudes = np.array([1, 1])
locations = np.array([[-0.3, -0.7], [0.2, 0.6]])
wave_number = 0.5
point_scat = PointScatteringProblem(wave_number, amplitudes, locations)

# far field computation
box_size = 2  # locations should belong to (-# box_size/2,box_size/2)
num_frequencies = 10000
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.linalg.norm(k)/(2*wave_number)) for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.linalg.norm(k)/(2*wave_number)) for k in frequencies])

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

# computation of BLASSO estimator
reg_param = 0.1
num_iter = 2

obs = np.concatenate([np.real(far_field), -np.imag(far_field)])
estimated_measure = solve_blasso(frequencies, obs, reg_param, num_iter, box_size)

print("true amplitudes:")
print(measure.amplitudes)

print("true locations:")
print(measure.locations)

print("estimated amplitudes:")
print(estimated_measure.amplitudes)

print("estimated locations:")
print(estimated_measure.locations)


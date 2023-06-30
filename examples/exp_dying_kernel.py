import numpy as np

from itertools import product

from pointscat.forward_problem import PointScatteringProblem
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure


np.random.seed(0)

amplitudes = np.array([0.5, 2, 1])
locations = np.array([[-3.3, -3.7], [-2.8, 3.5], [3.2, 2.6]])
wave_number = 1
point_scat = PointScatteringProblem(locations, amplitudes, wave_number)

box_size = 10  # locations should belong to (-box_size/2,box_size/2)
num_frequencies = 50
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.linalg.norm(k)/(2*wave_number))
                            for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.linalg.norm(k)/(2*wave_number))
                                   for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)
obs = np.concatenate([np.real(far_field), -np.imag(far_field)])

x_tab = np.linspace(-box_size/2, box_size/2, 3)
init_locations = np.array([[x_1, x_2] for (x_1, x_2) in product(x_tab, x_tab)])
init_amplitudes = np.ones(len(init_locations))
estimated_measure = DiscreteMeasure(init_locations, init_amplitudes)

state = estimated_measure.perform_nonlinear_sliding(incident_angles, observation_directions, obs, wave_number, box_size,
                                                    tol_locations=0.1, tol_amplitudes=0.01)

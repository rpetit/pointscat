import numpy as np
import matplotlib.pyplot as plt

from pointscat.forward_problem import PointScatteringProblem, compute_far_field
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
amplitudes = np.array([1, 1])
ref_locations = np.array([[-0.5, 0.0], [0.5, 0.0]])
wave_number = 1

box_size = 5

sep_dist_tab = [2, 1.5, 1, 0.5, 0.2]

for i in range(len(sep_dist_tab)):
    sep_dist = sep_dist_tab[i]
    locations = sep_dist * ref_locations

    point_scat = PointScatteringProblem(locations, amplitudes, wave_number)
    measure = DiscreteMeasure(locations, amplitudes)

    # far field computation
    num_frequencies = 15
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

    # parameters
    reg_param_lin = 0.5
    reg_param_nonlin = 0.001
    tol_locations = 1e-3
    tol_amplitudes = 0.1

    # computation of the BLASSO estimator under Born approx
    num_iter = 10

    estimated_measure = solve_blasso(frequencies, obs, reg_param_lin, num_iter, box_size,
                                     tol_locations=tol_locations, tol_amplitudes=tol_amplitudes)

    # display output of linear step
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
    ax.set_zlim(0, 2.1)

    plt.show()

    print("nonlinear sliding...")
    state = estimated_measure.perform_nonlinear_sliding(incident_angles, observation_directions,
                                                        obs, wave_number,
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
    ax.set_zlim(0, 2.1)

    plt.show()

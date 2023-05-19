import numpy as np
import matplotlib.pyplot as plt

from pointscat import *


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['+Computer Modern'],
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

wave_number = 2
amplitudes = np.array([3, 3, 3])

num_draws = 500  # number of configuration draws
eval_grid_size = 100  # number of indicent angles and observations directions
num_scatterers = len(amplitudes)

incident_angles = np.linspace(0, 2 * np.pi, eval_grid_size)
observation_directions = np.linspace(0, 2 * np.pi, eval_grid_size)

min_sep_dist_tab = []
smallest_sval_tab = []
one_norm_minus_id_tab = []
rel_err_tab = []

for i in range(num_draws):
    # drawing of the locations of the scatterers
    locations = 10 * np.random.random((num_scatterers, 2)) - 1

    # minimal separation distance computation
    min_sep_dist = np.linalg.norm(locations[0] - locations[1])
    min_sep_dist = min(min_sep_dist, np.linalg.norm(locations[0] - locations[2]))
    min_sep_dist = min(min_sep_dist, np.linalg.norm(locations[1] - locations[2]))
    min_sep_dist_tab.append(min_sep_dist)

    point_scat = PointScatteringProblem(wave_number, amplitudes, locations)

    # computation of Foldy matrix and related quantities
    smallest_sval, one_norm_minus_id = point_scat.compute_foldy_matrix(output=True)
    smallest_sval_tab.append(smallest_sval)
    one_norm_minus_id_tab.append(one_norm_minus_id)

    # far field computation (with and without Born approximation)
    far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)
    far_field = point_scat.compute_far_field(incident_angles, observation_directions)

    # computation of the relative error on the (discretized) far field
    rel_err = np.linalg.norm(far_field - far_field_born) / np.linalg.norm(far_field)
    rel_err_tab.append(rel_err)

# computation of the theoretical upper bound on the error
min_sep_dist_grid = np.linspace(np.min(min_sep_dist_tab), np.max(min_sep_dist_tab), 100)
aux = wave_number ** 2 * num_scatterers * np.max(amplitudes) / (4 * np.pi)
th_bound_grid = aux * (aux / min_sep_dist_grid) / (1 - aux / min_sep_dist_grid)

# plot relative error with respect to minimal separation distance
# TODO: fix because of change in compute_far_field
fig, axs = plt.subplots(3, 1, figsize=(7, 21))

axs[0].scatter(min_sep_dist_tab, rel_err_tab, marker='.')
# axs[0].plot(min_sep_dist_grid, th_bound_grid, color='red')

axs[0].set_xlabel('minimal separation distance')
axs[0].set_ylabel('relative error')
axs[0].set_title('relative error on the far field pattern')

axs[1].scatter(min_sep_dist_tab, smallest_sval_tab, marker='.')
axs[1].hlines(0, np.min(min_sep_dist_tab), np.max(min_sep_dist_tab), linestyle='--', color='black')

axs[1].set_xlabel('minimal separation distance')
axs[1].set_ylabel('smallest singular value')
axs[1].set_title('smallest singular value of Foldy matrix')

axs[2].scatter(min_sep_dist_tab, one_norm_minus_id_tab, marker='.')
axs[2].hlines(1, np.min(min_sep_dist_tab), np.max(min_sep_dist_tab), linestyle='--', color='black')

axs[2].set_xlabel('minimal separation distance')
axs[2].set_ylabel('one-norm')
axs[2].set_title('one-norm of Foldy matrix minus identity')

fig.suptitle('validity of Born approximation')
fig.tight_layout()

plt.show()

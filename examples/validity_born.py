import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from pointscat import *


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 35})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12, 7))


def run_exp(params):
    dist_tab = params['dist_tab']
    wave_number = params['wave_number']
    amplitudes = np.array([1, 1])

    num_draws = 20  # number of configuration draws
    num_measurements = 100  # number of indicent angles and observations directions

    incident_angles = 2 * np.pi * np.random.random(num_measurements)
    observation_directions = 2 * np.pi * np.random.random(num_measurements)

    err_tab = []
    mean_err_tab = []
    std_err_tab = []

    naive_err_tab = []

    for i in range(len(dist_tab)):
        # drawing of the locations of the scatterers
        first_locations = 2 * np.random.random((num_draws, 2)) - 1
        angles = 2 * np.pi * np.random.random(num_draws)
        second_locations = first_locations + dist_tab[i] * np.stack([np.cos(angles), np.sin(angles)], axis=1)

        err_tab_i = []
        naive_err_tab_i = []

        for j in range(num_draws):
            locations = np.vstack([first_locations[j], second_locations[j]])
            assert np.isclose(np.linalg.norm(first_locations[j] - second_locations[j]), dist_tab[i])

            point_scat = PointScatteringProblem(locations, amplitudes, wave_number)

            # far field computation (with and without Born approximation)
            far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)
            far_field = point_scat.compute_far_field(incident_angles, observation_directions)

            # computation of the error on the (discretized) far field
            err = np.linalg.norm(far_field - far_field_born) / np.sqrt(num_measurements)
            err_tab_i.append(err)

            # computation of the naive upper bound on the error
            naive_err = (np.linalg.norm(far_field) + np.linalg.norm(far_field_born)) / np.sqrt(num_measurements)
            naive_err_tab_i.append(naive_err)

        err_tab.append(err_tab_i)
        mean_err_tab.append(np.mean(err_tab_i))
        std_err_tab.append(np.std(err_tab_i))

        naive_err_tab.append(np.mean(naive_err_tab_i))

    # computation of the theoretical upper bound on the error
    green_function_val = np.array([green_function(wave_number, np.array([0, 0]), dist_tab[i] * np.array([1, 0]))
                                   for i in range(len(dist_tab))])
    mean_amp = 0.5 * (amplitudes[0] + amplitudes[1])
    geom_mean_amp = np.sqrt(amplitudes[0] * amplitudes[1])
    aux = wave_number**2 * geom_mean_amp * green_function_val

    # th_bound = 2 * np.abs(aux) / np.abs(1 - aux**2) * (np.abs(aux) * mean_amp + geom_mean_amp)
    # th_bound = 2 * np.abs(aux) / (1 - np.abs(aux)**2) * (np.abs(aux) * mean_amp + geom_mean_amp)

    # plot relative error with respect to minimal separation distance
    # ax.plot(dist_tab, np.array(mean_err_tab),
    #         color=params['color'],
    #         label=r'$\kappa={}$'.format(wave_number))

    ax.plot(dist_tab, np.array(mean_err_tab),
            color=params['color'],
            label='true error')

    # plot naive upper bound on the error
    ax.plot(dist_tab, naive_err_tab,
            color=params['color'],
            linestyle='-.',
            label='naive bound')

    # plot "error bars"
    # ax.fill_between(dist_tab, np.array(mean_err_tab) - 3*np.array(std_err_tab),
    #                 np.array(mean_err_tab) + 3*np.array(std_err_tab),
    #                 color=params['color'], alpha=0.2)

    # # plot theoretical bound on the error
    # ax.plot(dist_tab, th_bound,
    #         color=params['color'],
    #         linestyle='--')

    # Weird stuff
    dist = dist_tab[np.argmin(np.abs(np.abs(aux)-1))]
    ax.vlines(dist, 10**(-4.5), 10**3, color='black', linestyles='--')
    print(np.abs(aux))


size_dist_tab = 300

# dist_tab = np.logspace(-3.5, 5, size_dist_tab)
# params = {'wave_number': 1e-1, 'color': 'blue', 'dist_tab': dist_tab}
# run_exp(params)

# dist_tab = np.logspace(-2.59, 5, size_dist_tab)
dist_tab = np.logspace(-4, 0, size_dist_tab)
params = {'wave_number': 1, 'color': 'red', 'dist_tab': dist_tab}
run_exp(params)

# dist_tab = np.logspace(1.6, 5, size_dist_tab)
# params = {'wave_number': 10, 'color': 'purple', 'dist_tab': dist_tab}
# run_exp(params)

# final plot formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\Delta$')
ax.set_ylabel('error')
ax.set_ylim(10**(-0.6), 10**1.4)
ax.tick_params(axis='x', which='major', pad=5)

ax.legend(bbox_to_anchor=(1.46, 1), borderaxespad=0, framealpha=1)

plt.savefig('born_err.pdf', dpi=300, bbox_inches='tight')
# plt.show()

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

wave_number = 1
fig, ax = plt.subplots(figsize=(13, 8))


def run_exp(params):
    if params['amplitude_choice'] == 'same':
        amplitude = params['amplitude']
        amplitudes = amplitude * np.ones(2)

    if params['amplitude_choice'] == 'different':
        amplitudes = params['amplitudes']

    num_draws = 10  # number of configuration draws
    num_measurements = 100  # number of indicent angles and observations directions

    incident_angles = 2 * np.pi * np.random.random(num_measurements)
    observation_directions = 2 * np.pi * np.random.random(num_measurements)

    dist_tab = np.linspace(0.1, 20, 100)
    err_tab = []
    mean_err_tab = []
    std_err_tab = []

    for i in range(len(dist_tab)):
        # drawing of the locations of the scatterers
        first_locations = 2 * np.random.random((num_draws, 2)) - 1
        angles = 2 * np.pi * np.random.random(num_draws)
        second_locations = first_locations + dist_tab[i] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        err_tab_i = []

        for j in range(num_draws):
            locations = np.vstack([first_locations[j], second_locations[j]])
            assert np.isclose(np.linalg.norm(first_locations[j] - second_locations[j]), dist_tab[i])

            point_scat = PointScatteringProblem(locations, amplitudes, wave_number)

            # far field computation (with and without Born approximation)
            far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)
            far_field = point_scat.compute_far_field(incident_angles, observation_directions)

            # computation of the relative error on the (discretized) far field
            err = np.linalg.norm(far_field - far_field_born) / np.sqrt(num_measurements) * (2*np.pi)
            err_tab_i.append(err)

        err_tab.append(err_tab_i)
        mean_err_tab.append(np.mean(err_tab_i))
        std_err_tab.append(np.std(err_tab_i))

    # computation of the theoretical upper bound on the error
    green_function_val = np.array([green_function(wave_number, np.array([0, 0]), dist_tab[i] * np.array([1, 0]))
                                   for i in range(len(dist_tab))])
    mean_amp = 0.5 * (amplitudes[0] + amplitudes[1])
    geom_mean_amp = np.sqrt(amplitudes[0] * amplitudes[1])
    aux = wave_number**2 * geom_mean_amp * green_function_val
    th_bound = 4*np.pi * np.abs(aux) / np.abs(1 - aux**2) * (np.abs(aux) * mean_amp + geom_mean_amp)
    # TODO: investigate
    # th_bound_bis = 2 * np.abs(aux) / (1-np.abs(aux))

    # plot relative error with respect to minimal separation distance
    ax.plot(dist_tab, mean_err_tab,
            color=params['color'],
            label=r'emp. err. ($a_1={},~a_2={}$)'.format(amplitudes[0], amplitudes[1]))

    ax.fill_between(dist_tab, np.array(mean_err_tab) - 3*np.array(std_err_tab),
                    np.array(mean_err_tab) + 3*np.array(std_err_tab),
                    color=params['color'], alpha=0.2)

    ax.plot(dist_tab, th_bound,
            color=params['color'],
            linestyle='--',
            label=r'th. bound. ($a_1={},~a_2={}$)'.format(amplitudes[0], amplitudes[1]))

    # ax.plot(dist_tab, th_bound_bis, color='green', linestyle='--', label='theoretical bound bis')
    # axs[0].set_yscale('log')


params = {'amplitude_choice': 'same', 'amplitude': 1, 'color': 'red'}
run_exp(params)

params = {'amplitude_choice': 'same', 'amplitude': 3, 'color': 'blue'}
run_exp(params)

params = {'amplitude_choice': 'same', 'amplitude': 0.1, 'color': 'green'}
run_exp(params)

params = {'amplitude_choice': 'different', 'amplitudes': np.array([1, 2]), 'color': 'purple'}
run_exp(params)

params = {'amplitude_choice': 'different', 'amplitudes': np.array([0.1, 1]), 'color': 'black'}
run_exp(params)

# final plot formatting
ax.set_yscale('log')
ax.set_xlabel('distance')
ax.set_ylabel('relative error')
ax.set_title('relative error on the far field pattern')

ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

fig.suptitle('validity of Born approximation')
fig.tight_layout()

plt.show()

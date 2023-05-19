import numpy as np
import matplotlib.pyplot as plt

from pointscat.inverse_problem import frequency_grid, trigo_poly, DiscreteMeasure, solve_blasso


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['+Computer Modern'],
    'font.size': 20,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

np.random.seed(0)

# TODO: test stopping criterion based on dual gap

amplitudes = np.array([1, 2])
locations = np.array([[0.45, 0.5], [0.55, 0.5]])
unknown_measure = DiscreteMeasure(locations, amplitudes)

frequencies = frequency_grid(2)
observations = unknown_measure.compute_fourier_transform(frequencies)
fourier_series = lambda x: trigo_poly(x, frequencies, observations)

std_noise = 1.0
noise = np.random.normal(0, std_noise, size=observations.shape)
noisy_observations = observations + noise
noisy_fourier_series = lambda x: trigo_poly(x, frequencies, noisy_observations)

reg_param = 0.25 * np.sqrt(2 * np.log(len(observations))) * std_noise
num_iter = 2
estimated_measure = solve_blasso(frequencies, observations, reg_param, num_iter)

tab = np.linspace(0, 1, 100)
grid = np.array([[[tab[i], tab[j]] for j in range(len(tab))] for i in range(len(tab))])
f_grid = np.array([[fourier_series(grid[i, j]) for j in range(len(tab))] for i in range(len(tab))])
noisy_f_grid = np.array([[noisy_fourier_series(grid[i, j]) for j in range(len(tab))] for i in range(len(tab))])

v_abs_max = max(np.max(np.abs(f_grid)), np.max(np.abs(noisy_f_grid)))

colormap = 'bwr'

fig = plt.figure(figsize=plt.figaspect(3.))

ax = fig.add_subplot(3, 1, 1, projection='3d')

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

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_title('Unknown and \nestimated measures')

ax = fig.add_subplot(3, 1, 2)
im = ax.imshow(f_grid.T, cmap=colormap, vmin=-v_abs_max, vmax=v_abs_max, origin='lower')
ax.axis('off')
ax.set_title('Noiseless observations')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = fig.add_subplot(3, 1, 3)
im = ax.imshow(noisy_f_grid.T, cmap=colormap, vmin=-v_abs_max, vmax=v_abs_max, origin='lower')
ax.axis('off')
ax.set_title('Noisy observations')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()

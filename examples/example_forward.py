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


# setting problem
amplitudes = 0.5 * np.array([3, 3, 3])
locations = 0.5 * np.array([[-1.6, -0.2], [2.2, 1.6], [2.4, -1.7]])
# amplitudes = np.array([3, 3, 3])
# locations = np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
wave_number = 2

point_scat = PointScatteringProblem(locations, amplitudes, wave_number)

# defining evaluation grid
num_points = 200
x_tab = np.linspace(-6, 6, num_points)
X, Y = np.meshgrid(x_tab, x_tab)
x_grid = np.vstack([X.flatten(), Y.flatten()]).T

# incident wave and total field computation
incident_angle = np.pi/4

incident_wave_vals = np.exp(1j * wave_number * np.sum(angle_to_vec(incident_angle)[np.newaxis, :] * x_grid, axis=1))
incident_wave = incident_wave_vals.reshape((num_points, num_points))

total_field_vals = point_scat.compute_total_field(incident_angle, x_grid)
total_field_vals_born = point_scat.compute_total_field(incident_angle, x_grid, born_approx=True)
total_field = total_field_vals.reshape((num_points, num_points))
total_field_born = total_field_vals_born.reshape((num_points, num_points))
diff_total_field = total_field_born - total_field

scattered_field = total_field - incident_wave
scattered_field_born = total_field_born - incident_wave
diff_scattered_field = scattered_field_born - scattered_field

# plot
# TODO: wrap into plot util functions?
# TODO: plot scatterers
colormap = 'bwr'
vmin = -2.2
vmax = 2.2

# plot incident wave
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

im = axs[0].imshow(np.real(incident_wave), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[0].axis('off')
axs[0].set_title('incident wave (real part)')
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

im = axs[1].imshow(np.imag(incident_wave), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[1].axis('off')
axs[1].set_title('incident wave (imag. part)')
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

# plot scattered field
fig, axs = plt.subplots(3, 2, figsize=(14, 21))

im = axs[0, 0].imshow(np.real(scattered_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[0, 0].axis('off')
axs[0, 0].set_title('scattered field (real part)')
fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

im = axs[0, 1].imshow(np.imag(scattered_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[0, 1].axis('off')
axs[0, 1].set_title('scattered field (imag. part)')
fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

# plot scattered field under Born approximation
im = axs[1, 0].imshow(np.real(scattered_field_born), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[1, 0].axis('off')
axs[1, 0].set_title('Born scattered field (real part)')
fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

im = axs[1, 1].imshow(np.imag(scattered_field_born), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[1, 1].axis('off')
axs[1, 1].set_title('Born scattered field (imag. part)')
fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

# plot difference
im = axs[2, 0].imshow(np.real(diff_scattered_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[2, 0].axis('off')
axs[2, 0].set_title('diff scattered field (real part)')
fig.colorbar(im, ax=axs[2, 0], fraction=0.046, pad=0.04)

im = axs[2, 1].imshow(np.imag(diff_scattered_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[2, 1].axis('off')
axs[2, 1].set_title('diff scattered field (imag. part)')
fig.colorbar(im, ax=axs[2, 1], fraction=0.046, pad=0.04)

# plot total field
fig, axs = plt.subplots(3, 2, figsize=(14, 21))

im = axs[0, 0].imshow(np.real(total_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[0, 0].axis('off')
axs[0, 0].set_title('total field (real part)')
fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

im = axs[0, 1].imshow(np.imag(total_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[0, 1].axis('off')
axs[0, 1].set_title('total field (imag. part)')
fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

# plot total field under Born approximation

im = axs[1, 0].imshow(np.real(total_field_born), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[1, 0].axis('off')
axs[1, 0].set_title('Born total field (real part)')
fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

im = axs[1, 1].imshow(np.imag(total_field_born), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[1, 1].axis('off')
axs[1, 1].set_title('Born total field (imag. part)')
fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

# plot difference
im = axs[2, 0].imshow(np.real(diff_total_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[2, 0].axis('off')
axs[2, 0].set_title('diff total field (real part)')
fig.colorbar(im, ax=axs[2, 0], fraction=0.046, pad=0.04)

im = axs[2, 1].imshow(np.imag(diff_total_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
axs[2, 1].axis('off')
axs[2, 1].set_title('diff total field (imag. part)')
fig.colorbar(im, ax=axs[2, 1], fraction=0.046, pad=0.04)

fig.subplots_adjust()
plt.show()

# far field computation and plot
# TODO: find better way to plot, far field projection?
# TODO: plot far field under Born approximation
# TODO: fix because of change in compute_far_field
# incident_angles = np.linspace(0, 2*np.pi, num_points)
# observation_directions = np.linspace(0, 2*np.pi, num_points)
# far_field_vals = point_scat.compute_far_field(incident_angles, observation_directions)
# far_field = far_field_vals.reshape((num_points, num_points))
#
# fig, axs = plt.subplots(1, 2, figsize=(14, 7))
# v_abs_max = np.max(np.abs(far_field))
# vmin = -v_abs_max
# vmax = v_abs_max
#
# im = axs[0].imshow(np.real(far_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
# axs[0].axis('off')
# axs[0].set_title('far field pattern (real part)')
# fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
#
# im = axs[1].imshow(np.imag(far_field), origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
# axs[1].axis('off')
# axs[1].set_title('far field pattern (imag. part)')
# fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
#
# fig.subplots_adjust()
# plt.show()

import numpy as np
import jax.numpy as jnp
import jaxopt

from pointscat.forward_problem import PointScatteringProblem
from pointscat.inverse_problem import unif_sample_disk, DiscreteMeasure, solve_blasso, compute_far_field


np.random.seed(0)  # fix random seed

# setting problem
amplitudes = np.array([1.2, 2.3, 1.9])
locations = np.array([[-0.25, 0.925], [-0.8, 0.0], [0.9, 0.65]])
wave_number = 1

point_scat = PointScatteringProblem(locations, amplitudes, wave_number)
measure = DiscreteMeasure(locations, amplitudes)

box_size = 5  # locations should belong to (-box_size/2,box_size/2)

# far field computation
num_frequencies = 30
cutoff_frequency = 2 * wave_number
frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)

incident_angles = np.array([np.pi + np.angle(k[0]+1j*k[1]) - np.arccos(np.linalg.norm(k)/(2*wave_number))
                            for k in frequencies])
observation_directions = np.array([np.angle(k[0]+1j*k[1]) + np.arccos(np.linalg.norm(k)/(2*wave_number))
                                   for k in frequencies])

far_field = point_scat.compute_far_field(incident_angles, observation_directions)
far_field_born = point_scat.compute_far_field(incident_angles, observation_directions, born_approx=True)

# observations
obs = np.concatenate([np.real(far_field), -np.imag(far_field)])
born_obs = np.concatenate([np.real(far_field_born), -np.imag(far_field_born)])

std_noise = 0.25
noise = np.random.normal(scale=std_noise, size=obs.shape)
noisy_obs = obs + noise

# parameters
reg_param = 3.0
tol_locations = 0.05
tol_amplitudes = 0.01

# computation of the BLASSO estimator under Born approx
num_iter = 10

print("computation of the BLASSO estimator under Born approx...")
estimated_measure = solve_blasso(frequencies, noisy_obs, reg_param, num_iter, box_size,
                                 tol_locations=tol_locations, tol_amplitudes=tol_amplitudes)

num_spikes = estimated_measure.num_spikes


def sliding_obj(x):
    # parse input vector
    amplitudes = x[:num_spikes]
    locations = jnp.reshape(x[num_spikes:], (num_spikes, 2))

    far_field = compute_far_field(locations, amplitudes, wave_number, incident_angles, observation_directions)
    image = jnp.concatenate([jnp.real(far_field), -jnp.imag(far_field)])  # TODO: fix ugly

    return jnp.sum((image - noisy_obs) ** 2) / 2 + reg_param * jnp.sum(jnp.abs(amplitudes))


# vector of initial parameters
amplitudes = jnp.array(estimated_measure.amplitudes, dtype='float64')
locations = jnp.array(estimated_measure.locations.flatten(), dtype='float64')
x_0 = jnp.concatenate([amplitudes, locations])

lower_bounds_amplitudes = jnp.where(amplitudes > 0, 0, -jnp.inf)
upper_bounds_amplitudes = jnp.where(amplitudes < 0, 0, jnp.inf)

bound_locations = box_size / 2 * jnp.ones(2 * num_spikes)
lower_bounds = jnp.concatenate([lower_bounds_amplitudes, -bound_locations])
upper_bounds = jnp.concatenate([upper_bounds_amplitudes, bound_locations])

solver = jaxopt.LBFGSB(fun=sliding_obj)
init_params = x_0
bounds = (lower_bounds, upper_bounds)
kwargs = {'bounds': bounds}

state = solver.init_state(init_params, **kwargs)
zero_step = solver._make_zero_step(init_params, state)
opt_step = solver.update(init_params, state, **kwargs)

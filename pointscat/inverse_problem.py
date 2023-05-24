import numpy as np
import jax.numpy as jnp
import jaxopt

from itertools import product
from celer import Lasso


def unif_sample_disk(num_samples, radius):
    """
    Draw num_samples points uniformly in the 2D ball centered at 0 with a given radius

    Parameters
    ----------
    num_samples: int
        The number of samples
    radius: float
        Radius of the ball within which to sample

    Returns
    -------
    np.array, shape (num_frequencies,2)
        The list of samples
    """
    # TODO: pass random seed as input
    assert isinstance(num_samples, int)
    assert radius > 0

    r_tab = np.sqrt(np.random.random(num_samples)) * radius
    theta_tab = np.random.random(num_samples) * 2*np.pi

    return np.stack([r_tab * np.cos(theta_tab), r_tab * np.sin(theta_tab)], axis=1)


def trigo_poly(x, frequencies, coefficients):
    """
    Evaluate the trigonometric polynomial defined by a set of frequencies and coefficients at a point x

    Parameters
    ----------
    x: jnp.array, shape(2,)
        The point at which to evaluate the trigonometric polynomial
    frequencies: jnp.array, shape (M, 2)
        The set of frequencies
    coefficients: jnp.array, shape (2*M,)
        The coefficients associated to each frequency. First M coefficients are for cosines, last M for sines

    Returns
    -------
    jnp.float
        Value of the trigonometric polynomial at x
    """
    assert len(frequencies) * 2 == len(coefficients)

    dot_prod_array = jnp.dot(frequencies, x)
    res = jnp.sum(coefficients[:len(frequencies)] * jnp.cos(dot_prod_array))
    res += jnp.sum(coefficients[len(frequencies):] * jnp.sin(dot_prod_array))

    return res


def find_argmax_grid(f, box_size, grid_size):
    """
    Look for the maximizer of f on a regular grid of size grid_size * grid_size on [-box_size/2,box_size/2]^2

    Parameters
    ----------
    f: function
        Takes array inputs with shape (2,) TODO: allow (N, 2) inputs?
    box_size: int
        Size of the equare on which to optimize
    grid_size: int
        Total number of grid points is grid_size * grid_size

    Returns
    -------
    np.array, shape (2,)
        Maximizer of f on the grid
    """
    # construct an array of grid_size * grid_size equi-spaced points on [-box_size/2,box_size/2]^2
    x_tab = np.linspace(-box_size/2, box_size/2, grid_size)
    grid = np.array([[x_1, x_2] for (x_1, x_2) in product(x_tab, x_tab)])

    val_tab = np.array([f(x) for x in grid])  # evaluate function on grid

    return grid[np.argmax(val_tab)]


def find_argmax_abs(f, box_size, grid_size):
    # TODO: write doc
    # TODO: investigate why JAX is much slower to optimize than scipy
    # TODO: implement direct computation of the gradient
    def abs_f(x):
        return np.abs(f(x))

    argmax_abs_grid = find_argmax_grid(abs_f, box_size, grid_size)
    sign = jnp.sign(f(argmax_abs_grid))

    def signed_f(x):
        return -sign * f(x)

    solver = jaxopt.ScipyBoundedMinimize(fun=signed_f, method="l-bfgs-b")
    params, state = solver.run(argmax_abs_grid, bounds=(-box_size/2 * jnp.ones(2), box_size/2 * jnp.ones(2)))

    return params


def compute_fourier_transform(locations, amplitudes, frequencies):
    """
    Compute the Fourier transform of a discrete measure at a given set of frequencies.

    To avoid dealing with complex numbers, we output the real and imaginary parts of the Fourier coefficients

    Parameters
    ----------
    locations: np.array or jnp.array, shape (N, 2)
        Locations of the spikes
    amplitudes: np.array or jnp.array, shape (M,)
        Weights associated to each spike
    frequencies: np.array or jnp.array, shape (M, 2)
        Frequencies at which the Fourier transform is computed

    Returns
    -------
    jnp.array, shape (2M,)
        Fourier coefficients of the measure at each frequency. The 2*M-th coordinate is the real part of the M-th
        Fourier coefficient, and the 2*M+1-th coordinate the imaginary part
    """
    # TODO: check that frequencies have integer coordinates?
    assert frequencies.ndim == 2 and frequencies.shape[1] == 2
    dot_prod_array = jnp.dot(frequencies, locations.T)
    ft_real = jnp.sum(amplitudes[jnp.newaxis, :] * jnp.cos(dot_prod_array), axis=1)
    ft_imag = jnp.sum(amplitudes[jnp.newaxis, :] * jnp.sin(dot_prod_array), axis=1)

    return jnp.concatenate([ft_real, ft_imag])


# TODO: deal with spikes at same location
# TODO: deal with representation of locations (points on the torus)
# TODO: deal with complex amplitudes (sklearn and celer do not like complex numbers)?
class DiscreteMeasure:
    """
    Discrete measure class

    Attributes
    ----------
    locations: np.array, shape (N, 2)
        The locations of the spikes
    amplitudes: np.array, shape (N,)
        The weight associated to each spike. None if zero measure
    """

    # TODO: write doc about setters and getters
    def __init__(self, locations, amplitudes):
        assert amplitudes.ndim == 1
        assert locations.shape == (len(amplitudes), 2)

        self.locations = locations
        self.amplitudes = amplitudes

    @property
    def num_spikes(self):
        return len(self.amplitudes)

    def add_spike(self, new_location):
        """
        Add a new spike with zero amplitude

        Parameters
        ----------
        new_location: np.array, shape (2,)
            Location of the new spike
        """
        self.locations = np.vstack((self.locations, new_location))
        self.amplitudes = np.append(self.amplitudes, 0)

    def compute_fourier_transform(self, frequencies):
        """
        See compute_fourier_transform

        Parameters
        ----------
        frequencies: np.array or jnp.array, shape (M, 2)
            Frequencies at which the Fourier transform is computed

        Returns
        -------
        jnp.array, shape (2M,)
            Fourier coefficients of the measure at each frequency. The 2*M-th coordinate is the real part of the M-th
            Fourier coefficient, and the 2*M+1-th coordinate the imaginary part
        """
        return compute_fourier_transform(self.locations, self.amplitudes, frequencies)

    def fit_weights(self, frequencies, observations, reg_param, tol_factor=1e-4):
        # TODO: write doc
        # TODO: remove spikes with amplitude below some threshold
        dot_prod_array = np.dot(frequencies, self.locations.T)
        measurement_mat_real = np.cos(dot_prod_array)
        measurement_mat_imag = np.sin(dot_prod_array)
        measurement_mat = np.vstack([measurement_mat_real, measurement_mat_imag])

        tol = tol_factor * np.linalg.norm(observations) ** 2 / observations.size

        lasso = Lasso(alpha=reg_param/observations.size, fit_intercept=False, tol=tol)
        lasso.fit(measurement_mat, observations)

        self.amplitudes = lasso.coef_

    def perform_sliding(self, frequencies, observations, reg_param, box_size):
        # TODO: implement spike merging
        # TODO: write doc
        num_spikes = self.num_spikes

        def sliding_obj(x):
            # parse input vector
            amplitudes = x[:num_spikes]
            locations = jnp.reshape(x[num_spikes:], (num_spikes, 2))
            ft = compute_fourier_transform(locations, amplitudes, frequencies)

            return jnp.sum((ft - observations)**2) / 2 + reg_param * jnp.sum(jnp.abs(amplitudes))

        # vector of initial parameters
        # TODO: fix ugly conversion
        x_0 = jnp.concatenate([jnp.array(self.amplitudes, dtype='float32'),
                               jnp.array(self.locations.flatten(), dtype='float32')])

        bounds = jnp.concatenate([jnp.inf * jnp.ones(num_spikes), box_size/2 * jnp.ones(2*num_spikes)])

        solver = jaxopt.ScipyBoundedMinimize(fun=sliding_obj, method="l-bfgs-b")
        params, state = solver.run(x_0, bounds=(-bounds, bounds))

        new_amplitudes = params[:self.num_spikes]
        new_locations = np.reshape(params[self.num_spikes:], (self.num_spikes, 2))
        self.amplitudes = new_amplitudes
        self.locations = new_locations


def zero_measure():
    """
    Construct zero measure

    Returns
    -------
    DiscreteMeasure
        Zero measure
    """
    locations = np.empty((0, 2))
    amplitudes = np.empty(0)
    return DiscreteMeasure(locations, amplitudes)


def solve_blasso(frequencies, observations, reg_param, num_iter, box_size, grid_size=100):
    measure = zero_measure()

    for i in range(num_iter):
        residual = measure.compute_fourier_transform(frequencies) - observations

        def eta(x):
            return trigo_poly(x, frequencies, residual)

        new_location = find_argmax_abs(eta, box_size, grid_size)
        measure.add_spike(new_location)

        measure.fit_weights(frequencies, observations, reg_param)
        measure.perform_sliding(frequencies, observations, reg_param, box_size)

    return measure

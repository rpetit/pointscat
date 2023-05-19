import numpy as np
import jax.numpy as jnp
import jaxopt

from itertools import product
from celer import Lasso


def frequency_grid(cutoff_frequency):
    """
    List all 2D integer-valued vectors whose coordinates are non greater than the cutoff frequency

    Parameters
    ----------
    cutoff_frequency: int
        The cutoff frequency

    Returns
    -------
    np.array, shape ((cutoff_frequency+1)**2,2)
        The list of 2D frequency vectors
    """

    # TODO: find better name?
    assert isinstance(cutoff_frequency, int)

    f_tab = np.arange(cutoff_frequency + 1)
    return np.array([[f_1, f_2] for (f_1, f_2) in product(f_tab, f_tab)])


def trigo_poly(x, frequencies, coefficients):
    """
    Evaluate the trigonometric polynomial defined by a set of frequencies and coefficients at a point x

    Parameters
    ----------
    x: jnp.array, shape(2,)
        The point at which to evaluate the trigonometric polynomial
    frequencies: jnp.array, shape (N, 2)
        The set of frequencies
    coefficients: jnp.array, shape (2*N,)
        The coefficients associated to each frequency. First N coefficients are for cosines, last N for sines

    Returns
    -------
    jnp.float
        Value of the trigonometric polynomial at x
    """
    # TODO: test
    assert len(frequencies) * 2 == len(coefficients)

    dot_prod_array = jnp.dot(frequencies, x)
    res = jnp.sum(coefficients[:len(frequencies)] * jnp.cos(2*jnp.pi * dot_prod_array))
    res += jnp.sum(coefficients[len(frequencies):] * jnp.sin(2*jnp.pi * dot_prod_array))

    return res


def find_argmax_grid(f, grid_size):
    """
    Look for the maximizer of f on a regular grid of size grid_size * grid_size on the torus

    Parameters
    ----------
    f: function
        Takes array inputs with shape (2,)
    grid_size: int
        Total number of grid points is grid_size * grid_size

    Returns
    -------
    TODO: write
    """
    # construct an array of grid_size * grid_size equi-spaced points on the torus TODO: use product instead of meshgrid?
    x_tab, y_tab = np.linspace(0, 1, grid_size + 1)[:-1], np.linspace(0, 1, grid_size + 1)[:-1]
    x_grid, y_grid = np.meshgrid(x_tab, y_tab)
    grid = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)

    val_tab = np.array([f(grid[i]) for i in range(grid_size * grid_size)])  # evaluate function on grid

    return grid[np.argmax(val_tab)]


def find_argmax_abs(f, grid_size):
    # TODO: write doc
    # TODO: investigate why JAX is much slower to optimize than scipy
    # TODO: implement direct computation of the gradient
    def abs_f(x):
        return np.abs(f(x))

    argmax_abs_grid = find_argmax_grid(abs_f, grid_size)
    sign = jnp.sign(f(argmax_abs_grid))

    def signed_f(x):
        return -sign * f(x)

    solver = jaxopt.BFGS(signed_f)
    params, state = solver.run(argmax_abs_grid)

    return params


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

        self._num_spikes = len(amplitudes)

        self.locations = locations
        self.amplitudes = amplitudes

    @property
    def num_spikes(self):
        return self._num_spikes

    @property
    def amplitudes(self):
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, a):
        assert a.shape == (self.num_spikes,)
        self._amplitudes = a

    @property
    def locations(self):
        return self._locations

    @locations.setter
    def locations(self, x):
        assert x.shape == (self.num_spikes, 2)
        self._locations = np.mod(x, 1)  # we work on the torus and hence always keep locations between 0 and 1

    def add_spike(self, new_location):
        """
        Add a new spike with zero amplitude

        Parameters
        ----------
        new_location: np.array, shape (2,)
            Location of the new spike
        """
        self._num_spikes += 1
        self.locations = np.vstack((self.locations, new_location))
        self.amplitudes = np.append(self.amplitudes, 0)

    def compute_fourier_transform(self, frequencies):
        """
        Compute the Fourier transform of a measure at a given set of frequencies.

        To avoid dealing with complex numbers, we output the real and imaginary parts of the Fourier coefficients

        Parameters
        ----------
        frequencies: np.array, shape (N, 2)
            Frequencies at which the Fourier transform is computed

        Returns
        -------
        np.array, shape (2N,)
            Fourier coefficients of the measure at each frequency. The 2*N-th coordinate is the real part of the N-th
            Fourier coefficient, and the 2*N+1-th coordinate the imaginary part
        """
        # TODO: check that frequencies have integer coordinates?
        assert frequencies.ndim == 2 and frequencies.shape[1] == 2
        dot_prod_array = np.dot(frequencies, self.locations.T)
        ft_real = np.sum(self.amplitudes[np.newaxis, :] * np.cos(2*np.pi * dot_prod_array), axis=1)
        ft_imag = np.sum(self.amplitudes[np.newaxis, :] * np.sin(2*np.pi * dot_prod_array), axis=1)

        return np.concatenate([ft_real, ft_imag])

    def fit_weights(self, frequencies, observations, reg_param, tol_factor=1e-4):
        # TODO: write doc
        # TODO: remove spikes with amplitude below some threshold
        dot_prod_array = np.dot(frequencies, self.locations.T)
        measurement_mat_real = np.cos(2*np.pi * dot_prod_array)
        measurement_mat_imag = np.sin(2*np.pi * dot_prod_array)
        measurement_mat = np.vstack([measurement_mat_real, measurement_mat_imag])

        tol = tol_factor * np.linalg.norm(observations) ** 2 / observations.size

        lasso = Lasso(alpha=reg_param/observations.size, fit_intercept=False, tol=tol)
        lasso.fit(measurement_mat, observations)

        self.amplitudes = lasso.coef_

    def perform_sliding(self, frequencies, observations, reg_param):
        # TODO: implement spike merging
        num_spikes = self.num_spikes

        def sliding_obj(x):
            # parse input vector
            amplitudes = x[:num_spikes]
            locations = jnp.reshape(x[num_spikes:], (num_spikes, 2))

            # compute Fourier transform
            dot_prod_array = jnp.dot(frequencies, locations.T)
            ft_real = jnp.sum(amplitudes[jnp.newaxis, :] * jnp.cos(2*jnp.pi * dot_prod_array), axis=1)
            ft_imag = jnp.sum(amplitudes[jnp.newaxis, :] * jnp.sin(2*jnp.pi * dot_prod_array), axis=1)
            ft = jnp.concatenate([ft_real, ft_imag])

            return jnp.sum((ft - observations)**2) / 2 + reg_param * jnp.sum(jnp.abs(amplitudes))

        # vector of initial parameters
        # TODO: fix ugly conversion
        x_0 = jnp.concatenate([jnp.array(self.amplitudes, dtype='float32'),
                               jnp.array(self.locations.flatten(), dtype='float32')])

        solver = jaxopt.BFGS(sliding_obj)
        params, state = solver.run(x_0)

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


def solve_blasso(frequencies, observations, reg_param, num_iter, grid_size=100):
    measure = zero_measure()

    for i in range(num_iter):
        residual = measure.compute_fourier_transform(frequencies) - observations

        def eta(x):
            return trigo_poly(x, frequencies, residual)

        new_location = find_argmax_abs(eta, grid_size)
        measure.add_spike(new_location)

        measure.fit_weights(frequencies, observations, reg_param)
        measure.perform_sliding(frequencies, observations, reg_param)

    return measure

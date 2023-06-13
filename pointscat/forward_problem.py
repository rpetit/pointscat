import numpy as np
import jax.numpy as jnp

from .hankel import h0

from jax.config import config
config.update("jax_enable_x64", True)


def angle_to_vec(theta):
    """
    Convert a real number into the corresponding unit norm 2d vector

    Parameters
    ----------
    theta: float or array, shape (N,)

    Returns
    -------
    float or array, shape (N, 2)
    """
    if jnp.isscalar(theta):
        return jnp.array([jnp.cos(theta), jnp.sin(theta)])
    else:
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)


def green_function(wave_number, x, y):
    """
    Green function associated to the 2D Helmoltz equation

    Parameters
    ----------
    wave_number: float
        The wave number
    x: np.array, shape (2,)
        First evaluation point
    y: np.array, shape (2,)
        Second evaluation point

    Returns
    -------
    complex
        Value of the Green function at (x, y)
    """
    return 1j/4 * h0(wave_number * jnp.linalg.norm(x-y))


def compute_foldy_matrix(locations, amplitudes, wave_number):
    # TODO: write doc
    # TODO: test sign of off-diagonal coefficients (the minus was missing before, and tests still passed)
    num_scatterers = len(amplitudes)
    assert amplitudes.ndim == 1 and locations.shape == (num_scatterers, 2)

    foldy_mat = jnp.array([[-wave_number**2 * green_function(wave_number, locations[i], locations[j]) if i != j else 1
                            for j in range(num_scatterers)] for i in range(num_scatterers)], dtype='complex64')

    return foldy_mat


def solve_foldy_systems(locations, amplitudes, wave_number, incident_angles, born_approx=False):
    """
    Solve the Foldy system associated to a set of incident angles

    Parameters
    ----------
    locations: array, shape (N, 2)
        Locations of the scatterers
    amplitudes: array, shape (N,)
        Amplitude associated to each scatterer
    wave_number: float
        Wave number
    incident_angles: array, shape (M,)
        The set of angles defining the incident waves
    born_approx: bool
        Whether to use Born approximation

    Returns
    -------
    array, shape (N,M)
        The j-th column is the solution of the Foldy system associated to the incident wave defined by incident_angles[j]
    """
    # compute and inverse Foldy matrix
    foldy_matrix = compute_foldy_matrix(locations, amplitudes, wave_number)
    inv_foldy_matrix = jnp.linalg.inv(foldy_matrix)

    # compute the matrix whose (i,j)-th coefficient is the inner product between locations[i] and incident_angles[j]
    dot_prod = jnp.dot(locations, angle_to_vec(incident_angles).T)

    # compute right hand side of Foldy system
    right_hand_sides = jnp.exp(1j * wave_number * dot_prod)

    if born_approx:
        solutions = right_hand_sides
    else:
        solutions = jnp.dot(inv_foldy_matrix, right_hand_sides)

    return solutions


def compute_far_field(locations, amplitudes, wave_number, incident_angles, observations_directions, born_approx=False):
    # TODO: write doc
    foldy_sols = solve_foldy_systems(locations, amplitudes, wave_number, incident_angles, born_approx)

    # compute the matrix whose (i,j)-th coefficient is the inner product between locations[i]
    # and observations_directions[j]
    dot_prod = jnp.dot(locations, angle_to_vec(observations_directions).T)

    return jnp.sum(amplitudes[:, jnp.newaxis] * jnp.exp(-1j * wave_number * dot_prod) * foldy_sols, axis=0)


class PointScatteringProblem:

    def __init__(self, locations, amplitudes, wave_number):
        # TODO: write doc
        assert len(amplitudes) == len(locations)

        self.locations = locations
        self.amplitudes = amplitudes
        self.wave_number = wave_number

    @property
    def num_scatterers(self):
        return len(self.amplitudes)

    def solve_foldy_systems(self, incident_angles, born_approx=False):
        return solve_foldy_systems(self.locations, self.amplitudes, self.wave_number, incident_angles, born_approx)

    def compute_total_field(self, incident_angle, x, born_approx=False):
        """
        Computes the total field associated to some incident wave at a given point

        Parameters
        ----------
        incident_angle: float
            The incident angle (should be between 0 and 2*pi)
        x: np.array, shape (N, 2)
            The list of points at which the total field should be evaluated
        born_approx: bool
            Whether to use Born approximation, defaut False (no approximation)

        Returns
        -------
        np.array, shape (N,)
            Value of the total field associated to the incident wave with angle incident_angle evaluated at each x[i]
        """
        # TODO: allow to pass an array with shape (2,) as input for x?
        assert x.ndim == 2 and x.shape[1] == 2

        foldy_solution = self.solve_foldy_systems(jnp.array([incident_angle]), born_approx=born_approx)

        num_eval_points = len(x)
        res = np.array([np.exp(1j * self.wave_number * np.dot(angle_to_vec(incident_angle), x[i]))
                        for i in range(num_eval_points)])

        for i in range(num_eval_points):
            for j in range(self.num_scatterers):
                res[i] += self.amplitudes[j] * green_function(self.wave_number, self.locations[j], x[i]) * foldy_solution[j]

        return res

    def compute_far_field(self, incident_angles, observation_directions, born_approx=False):
        """
        Computes the far field pattern associated to a list of indicent angles and observation directions

        Parameters
        ----------
        incident_angles: np.array, shape (N,)
            The list of incident angles
        observation_directions: np.array, shape (N,)
            The list of observation directions
        born_approx: bool
            Whether to use Born approximation, defaut False (no approximation)

        Returns
        -------
        np.array, shape (N,)
            Values of the far field pattern for each incident angle and observation direction
        """
        return compute_far_field(self.locations, self.amplitudes, self.wave_number,
                                 incident_angles, observation_directions, born_approx=born_approx)

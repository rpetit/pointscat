import numpy as np

from scipy.optimize import minimize


# TODO: deal with spikes at same location
# TODO: deal with representation of locations (points on the torus)
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

    def __init__(self, locations, amplitudes):
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
        Compute the Fourier transform of the measure at a set of frequencies

        Parameters
        ----------
        frequencies: np.array, shape (N, 2)
            Frequencies at which the compute the Fourier transform

        Returns
        -------
        np.array, shape (N,)
            Fourier transform of the measure evaluated at each frequency
        """
        # TODO: check that frequencies have integer coordinates?
        assert frequencies.ndim == 2 and frequencies.shape[1] == 2
        dot_prod_array = np.dot(frequencies, self.locations.T)
        res = np.sum(self.amplitudes[np.newaxis, :] * np.exp(1j * 2*np.pi * dot_prod_array), axis=1)

        return res


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


def grid_search(f, grid_size):
    x_tab = np.linspace(0, 1, grid_size + 1)[:-1]
    val_tab = f(x_tab)
    return x_tab[np.argmax(val_tab)]


def find_argmax_abs(f, grid_size):
    def abs_f(x):
        return np.abs(f(x))

    argmax_abs_grid = grid_search(abs_f, grid_size)
    sign = np.sign(f(argmax_abs_grid))

    def sign_f(x):
        return -sign * f(x)

    res = minimize(sign_f, argmax_abs_grid, method='')

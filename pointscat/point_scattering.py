import numpy as np

from scipy.special import hankel1


def angle_to_vec(theta):
    return np.array([np.cos(theta), np.sin(theta)])


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
    return 1j/4 * hankel1(0, wave_number * np.linalg.norm(x-y))


class PointScatteringProblem:

    def __init__(self, wave_number, amplitudes, locations):
        # TODO: write doc
        assert len(amplitudes) == len(locations)

        self.wave_number = wave_number
        self.amplitudes = amplitudes
        self.locations = locations

        self.foldy_matrix = None

    @property
    def num_scatterers(self):
        return len(self.amplitudes)

    def compute_foldy_matrix(self):
        # TODO: write doc
        foldy_matrix = np.zeros((self.num_scatterers, self.num_scatterers), dtype='complex_')
        for i in range(self.num_scatterers):
            for j in range(self.num_scatterers):
                if i == j:
                    foldy_matrix[i, j] = 1
                else:
                    foldy_matrix[i, j] = self.wave_number * green_function(self.wave_number,
                                                                           self.locations[i],
                                                                           self.locations[j])

        self.foldy_matrix = foldy_matrix

    def solve_fold_system(self, incident_angle, born_approx=False):
        # TODO: write doc
        if not born_approx and self.foldy_matrix is None:
            self.compute_foldy_matrix()

        right_hand_side = np.array([np.exp(1j * self.wave_number * np.dot(angle_to_vec(incident_angle), self.locations[i]))
                                    for i in range(self.num_scatterers)])

        if born_approx:
            solution = right_hand_side
        else:
            solution = np.linalg.solve(self.foldy_matrix, right_hand_side)

        return solution

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
            Whether or not to use Born approximation, defaut False (no approximation)

        Returns
        -------
        np.array, shape (N,)
            Value of the total field associated to the incident wave with angle incident_angle evaluated at each x[i]
        """
        # TODO: allow to pass an array with shape (2,) as input for x?
        assert x.ndim == 2 and x.shape[1] == 2

        foldy_solution = self.solve_fold_system(incident_angle, born_approx=born_approx)

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
        x: np.array, shape (N,)
            The list of observation directions
        born_approx: bool
            Whether or not to use Born approximation, defaut False (no approximation)

        Returns
        -------
        np.array, shape (N,)
            Values of the far field pattern for each incident angle and observation direction
        """
        # TODO: allow to pass an array with shape (2,) as input for x?
        assert incident_angles.ndim == 1 and observation_directions.ndim == 1

        n, m = len(incident_angles), len(observation_directions)
        res = np.zeros((n, m), dtype='complex_')

        for i in range(n):
            foldy_solution = self.solve_fold_system(incident_angles[i], born_approx=born_approx)
            for j in range(m):
                for k in range(self.num_scatterers):
                    dot_aux = np.dot(angle_to_vec(observation_directions[j]), self.locations[k])
                    res[i, j] += self.amplitudes[k] * np.exp(-1j * self.wave_number * dot_aux) * foldy_solution[k]

        return res

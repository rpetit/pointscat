import numpy as np

from pointscat.inverse_problem import unif_sample_disk, trigo_poly, find_argmax_abs, find_argmax_grid
from pointscat.inverse_problem import DiscreteMeasure, zero_measure

from jax import config
config.update("jax_enable_x64", True)

# fix random feed for frequency sampling
np.random.seed(0)


def test_unif_sample_disk():
    radius = 2
    num_samples = 10
    samples = unif_sample_disk(num_samples, radius)

    assert samples.shape == (num_samples, 2)
    assert np.max(np.linalg.norm(samples, axis=1)) <= radius


def test_trigo_poly():
    frequencies = np.array([[0, 0], [0, 1], [1, 1]])
    coefficients = np.array([1, 1, -1, 2, 0, 3])

    # should be 1 + cos(x_2) - cos(x_1 + x_2) + 3 * sin(x_1 + x_2)
    def f(x):
        return trigo_poly(x, frequencies, coefficients)

    assert f(np.array([0, 0])) == 1
    assert f(np.array([np.pi/4, np.pi/4])) == 1 + np.sqrt(2)/2 + 3


def test_discrete_measure():
    # test zero measure and add spike
    measure = zero_measure()
    assert measure.num_spikes == 0

    new_location = np.array([0, 0])
    measure.add_spike(new_location)
    assert measure.num_spikes == 1

    # test constructor and add spike
    locations = np.array([[1, 2], [0, 0]])
    amplitudes = np.array([0, 2])

    measure = DiscreteMeasure(locations, amplitudes)
    assert measure.num_spikes == 2

    new_location = np.array([1, 1])
    measure.add_spike(new_location)
    assert measure.num_spikes == 3

    # test Fourier transform computation
    measure = zero_measure()
    frequencies = np.array([[0, 0], [1, -1]])
    ft = measure.compute_fourier_transform(frequencies)
    assert ft.shape == (4,)
    assert np.allclose(ft, 0)

    measure = DiscreteMeasure(np.array([[0, 1]]), np.array([2]))
    frequencies = np.array([[np.pi/2, np.pi/2], [np.pi/2, -np.pi/2]])
    ft = measure.compute_fourier_transform(frequencies)
    assert ft.shape == (4,)
    assert np.isclose(ft[0], 0)
    assert np.isclose(ft[1], 0)
    assert np.isclose(ft[2], 2)
    assert np.isclose(ft[3], -2)


def test_fit_weights():
    locations = np.array([[0, 0]])
    amplitudes = np.array([1])
    measure = DiscreteMeasure(locations, amplitudes)
    num_frequencies = 5
    cutoff_frequency = 1
    frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)
    observations = measure.compute_fourier_transform(frequencies)

    # change the amplitude after the FT computation to be sure fit_weights modifies it
    measure.amplitudes = np.array([0])
    assert measure.amplitudes[0] == 0

    reg_param = 1e-6
    measure.fit_weights(frequencies, observations, reg_param)
    assert np.isclose(measure.amplitudes[0], 1)


def test_find_argmax():
    frequencies = np.array([[0, 0], [1, 0], [1, 1]])
    coefficients = np.array([-1, 0, -1, 0, 2, 0])

    # should be -1 - cos(x_1+x_2) + 2*sin(x_1)
    def f(x):
        return trigo_poly(x, frequencies, coefficients)

    # test grid search (maximum of f should be 2)
    box_size = 2.1 * np.pi  # if equal to 2*np.pi, maximizer is found even for very coarse grids
    tol_1 = 1e-4
    tol_2 = 1e-10

    # grid is not fine enough
    argmax_1 = find_argmax_grid(f, box_size, 10)
    assert f(argmax_1) < 2 - tol_1

    # grid is fine enough
    argmax_2 = find_argmax_grid(f, box_size, 100)
    assert f(argmax_2) > 2 - tol_1

    # box is too small
    argmax_1 = find_argmax_grid(f, box_size/10, 100)
    assert f(argmax_1) < 2 - tol_1

    # test grid + local search (maximum of absolute value of f should be 4)
    argmax_abs_1, max_abs_1 = find_argmax_abs(f, box_size, 5)
    assert max_abs_1 > 4 - tol_2

    # test that multiplying f by -1 does not change the result
    def g(x):
        return -f(x)

    argmax_abs_2, max_abs_2 = find_argmax_abs(g, box_size, 5)
    assert max_abs_1 == max_abs_2


def test_sliding():
    locations = np.array([[0, 0]])
    amplitudes = np.array([1])
    measure = DiscreteMeasure(locations, amplitudes)
    num_frequencies = 5
    cutoff_frequency = 1
    frequencies = unif_sample_disk(num_frequencies, cutoff_frequency)
    observations = measure.compute_fourier_transform(frequencies)

    # change the amplitude after the FT computation to be sure the sliding modifies it
    measure.amplitudes = np.array([0.95])
    measure.locations = np.array([[0, 0]])

    reg_param = 1e-6
    box_size = 1
    measure.perform_sliding(frequencies, observations, reg_param, box_size)
    assert np.isclose(measure.amplitudes[0], 1)
    assert np.isclose(measure.locations[0, 0], 0)
    assert np.isclose(measure.locations[0, 1], 0)

    # change the location after the FT computation to be sure the sliding modifies it
    measure.amplitudes = np.array([0.9])
    measure.locations = np.array([[0.1, -0.1]])

    reg_param = 1e-6
    measure.perform_sliding(frequencies, observations, reg_param, box_size)
    assert np.isclose(measure.amplitudes[0], 1)
    assert np.isclose(measure.locations[0, 0], 0, atol=1e-5)
    assert np.isclose(measure.locations[0, 1], 0, atol=1e-5)

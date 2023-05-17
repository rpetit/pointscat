import numpy as np

from pointscat.blasso import frequency_grid, find_argmax_grid, find_argmax_abs, DiscreteMeasure, zero_measure


def test_frequency_grid():
    cutoff_frequency = 2
    frequencies = frequency_grid(cutoff_frequency)

    assert frequencies.shape == ((2*cutoff_frequency+1)**2, 2)
    np.testing.assert_array_equal(frequencies[0], np.array([-2, -2]))
    np.testing.assert_array_equal(frequencies[-1], np.array([2, 2]))


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

    measure = DiscreteMeasure(np.array([[0, 0]]), np.array([1]))
    frequencies = np.array([[1, 1], [1, -1]])
    ft = measure.compute_fourier_transform(frequencies)
    assert ft.shape == (4,)
    assert ft[0] == ft[1] == 1
    assert ft[2] == ft[3] == 0


def test_fit_weights():
    locations = np.array([[0, 0]])
    amplitudes = np.array([1])
    measure = DiscreteMeasure(locations, amplitudes)
    cutoff_frequency = 2
    frequencies = frequency_grid(cutoff_frequency)
    observations = measure.compute_fourier_transform(frequencies)

    # change the amplitude after the FT computation to be sure fit_weights modifies it
    measure.amplitudes = np.array([0])
    assert measure.amplitudes[0] == 0

    reg_param = 1e-6
    measure.fit_weights(frequencies, observations, reg_param)
    assert np.isclose(measure.amplitudes[0], 1)


def test_find_argmax():
    # test grid search
    def f(x):
        return np.cos(0.5 + 2 * np.pi * (x[0] + x[1])) - 1

    argmax = find_argmax_grid(f, 10)
    assert f(argmax) > -0.01

    # test grid + local search
    argmax_abs_1 = find_argmax_abs(f, 10)
    assert np.abs(f(argmax_abs_1)) > 0.999

    def g(x):
        return -f(x)

    argmax_abs_2 = find_argmax_abs(g, 10)
    assert np.abs(f(argmax_abs_1)) == np.abs(f(argmax_abs_2))


# def test_sliding():
#     locations = np.array([[0, 0]])
#     amplitudes = np.array([1])
#     measure = DiscreteMeasure(locations, amplitudes)
#     cutoff_frequency = 5
#     frequencies = frequency_grid(cutoff_frequency)
#     observations = measure.compute_fourier_transform(frequencies)
#
#     # change the amplitude after the FT computation to be sure the sliding modifies it
#     measure.amplitudes = np.array([0.95])
#     measure.locations = np.array([[0, 0]])
#
#     reg_param = 1e-6
#     measure.perform_sliding(frequencies, observations, reg_param)
#     assert np.isclose(measure.amplitudes[0], 1)
#     assert np.isclose(measure.locations[0, 0], 0, atol=1e-5)
#     assert np.isclose(measure.locations[0, 1], 0, atol=1e-5)
#
#     # change the location after the FT computation to be sure the sliding modifies it
#     measure.amplitudes = np.array([1])
#     measure.locations = np.array([[0.01, 0]])
#
#     reg_param = 1e-6
#     measure.perform_sliding(frequencies, observations, reg_param)
#     assert np.isclose(measure.amplitudes[0], 1)
#     assert np.isclose(measure.locations[0, 0], 0, atol=1e-5)
#     assert np.isclose(measure.locations[0, 1], 0, atol=1e-5)

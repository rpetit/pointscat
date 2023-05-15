import numpy as np

from pointscat.blasso import DiscreteMeasure, zero_measure, grid_search


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
    assert ft.shape == (2,)
    assert np.allclose(ft, 0)

    measure = DiscreteMeasure(np.array([[0, 0]]), np.array([1]))
    frequencies = np.array([[1, 1], [1, -1]])
    ft = measure.compute_fourier_transform(frequencies)
    assert ft.shape == (2,)
    assert ft[0] == ft[1] == 1


def test_find_argmax():
    f = lambda x: np.cos(0.5 + 2 * np.pi * x)
    argmax = grid_search(f, 100)
    assert np.isclose(f(argmax), 1)

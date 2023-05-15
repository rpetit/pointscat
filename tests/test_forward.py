import numpy as np


from pointscat.forward_problem import green_function, PointScatteringProblem


def test_green_function():
    x = np.array([0, 0])
    y = np.array([0, 0])
    wave_number = 1

    val = green_function(wave_number, x, y)

    assert np.iscomplexobj(val)
    # TODO: find clever test


def test_point_scattering_problem():
    amplitudes = np.array([1])
    locations = np.array([[0, 0]])
    wave_number = 1

    # test class constructor
    scat_prob = PointScatteringProblem(wave_number, amplitudes, locations)

    assert scat_prob.num_scatterers == 1

    # test Foldy matrix computation
    scat_prob.compute_foldy_matrix()

    assert scat_prob.foldy_matrix.shape == (1, 1)
    assert np.iscomplexobj(scat_prob.foldy_matrix)

    # test Foldy system resolution (with and without Born approximation)
    foldy_sol = scat_prob.solve_fold_system(0)
    foldy_sol_born = scat_prob.solve_fold_system(0, born_approx=True)

    assert len(foldy_sol) == 1 and len(foldy_sol_born) == 1
    assert foldy_sol[0] == 1 and foldy_sol_born[0] == 1

    # test total field computation
    x = np.array([[0.2, 0.2], [0.2, 0.2]])
    incident_angle = 0
    vals_total_field = scat_prob.compute_total_field(incident_angle, x)

    assert np.iscomplexobj(vals_total_field)
    assert len(vals_total_field) == 2
    assert vals_total_field[0] == vals_total_field[1]  # evaluation at same location

    # far field computation
    incident_angles = np.array([0])
    observation_directions = np.array([0])
    far_field = scat_prob.compute_far_field(incident_angles, observation_directions)

    assert np.iscomplexobj(far_field)
    assert far_field.shape == (1, 1)
    assert far_field[0, 0] == np.dot(amplitudes, foldy_sol)  # check expression of far field at 0 with incident angle

    # TODO: find clever test for Born approximation

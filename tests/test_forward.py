import jax.numpy as jnp

from pointscat.forward_problem import angle_to_vec, green_function
from pointscat.forward_problem import compute_foldy_matrix, solve_foldy_systems, compute_far_field
from pointscat.forward_problem import PointScatteringProblem

from jax.config import config
config.update("jax_enable_x64", True)


def test_angle_to_vec():
    theta = jnp.linspace(0, 1, 10)
    vecs = angle_to_vec(theta)

    assert vecs.shape == (10, 2)
    assert jnp.allclose(jnp.linalg.norm(vecs, axis=1), 1)


def test_green_function():
    x = jnp.array([0, 0])
    y = jnp.array([0, 0])
    wave_number = 1

    val = green_function(wave_number, x, y)

    assert jnp.iscomplexobj(val)
    # TODO: find clever test


def test_foldy_matrix():
    amplitudes = jnp.array([1, 2, 0.5])
    locations = jnp.array([[0, 0], [-1, 1], [0, 1]])
    wave_number = 1

    foldy_mat = compute_foldy_matrix(locations, amplitudes, wave_number)

    assert foldy_mat.shape == (len(amplitudes), len(amplitudes))
    assert jnp.iscomplexobj(foldy_mat)
    assert jnp.allclose(jnp.diagonal(foldy_mat), 1)


def test_foldy_resolution():
    amplitudes = jnp.array([1, 2, 0.5])
    locations = jnp.array([[0, 0], [-1, 1], [0, 1]])
    wave_number = 1

    num_obs = 10
    incident_angles = jnp.linspace(0, 2*jnp.pi, num_obs)
    incident_angles_vec = angle_to_vec(incident_angles)

    foldy_mat = compute_foldy_matrix(locations, amplitudes, wave_number)
    right_hand_sides = jnp.exp(1j * wave_number * jnp.array([[jnp.dot(locations[i], incident_angles_vec[j])
                                                              for j in range(num_obs)] for i in range(len(locations))]))
    foldy_sols = solve_foldy_systems(locations, amplitudes, wave_number, incident_angles)

    assert foldy_sols.shape == (len(amplitudes), num_obs)
    assert jnp.allclose(jnp.dot(foldy_mat, foldy_sols), right_hand_sides)


def test_far_field():
    amplitudes = jnp.array([1, 2, 0.5])
    locations = jnp.array([[0, 0], [-1, 1], [0, 1]])
    wave_number = 1

    num_obs = 10
    incident_angles = jnp.linspace(0, 2 * jnp.pi, num_obs)
    observation_directions = jnp.linspace(0, 2*jnp.pi, num_obs)
    incident_angles_vec = angle_to_vec(incident_angles)

    far_field = compute_far_field(locations, amplitudes, wave_number, incident_angles, observation_directions)

    assert far_field.shape == (num_obs,)

    # incident_angles[0] and incident_angles[-1] are equal modulo 2*pi (same thing for observation_directions)
    assert jnp.isclose(far_field[0], far_field[-1])


# TODO: amplitudes were not taken into account in Foldy matrix computation, implement test to check things are now ok
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
    foldy_sol = scat_prob.solve_foldy_system(0)
    foldy_sol_born = scat_prob.solve_foldy_system(0, born_approx=True)

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
    assert far_field.shape == (1,)
    assert far_field[0] == np.dot(amplitudes, foldy_sol)  # check expression of far field at 0 with incident angle

    # TODO: find clever test for Born approximation

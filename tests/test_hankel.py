import numpy as np

from scipy.special import jv, yv, hankel1
from pointscat.hankel import j0, y0, h0


def test_j0():
    x = np.linspace(0, 50, 10000)
    y_scipy = jv(0, x)
    y_jax = j0(x)
    tol = 1e-14
    assert np.max(np.abs(y_scipy - y_jax)) < tol


def test_y0():
    x = np.linspace(1e-15, 50, 10000)
    y_scipy = yv(0, np.abs(x))
    y_jax = y0(x)
    tol = 5e-10  # TODO: investigate why it fails for lower precisions
    assert np.max(np.abs(y_scipy - y_jax)) < tol


def test_h0():
    x = np.linspace(1e-15, 50, 10000)
    y_scipy = hankel1(0, x)
    y_jax = h0(x, epsilon=0)
    tol = 1e-8
    assert np.max(np.abs(y_scipy - y_jax) / np.abs(y_scipy)) < tol

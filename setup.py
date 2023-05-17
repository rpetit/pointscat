from setuptools import setup


setup(name='pointscat',
      install_requires=['numpy', 'scipy', 'celer', 'jax'],
      description="Localization of point scatterers",
      packages=['pointscat']
      )

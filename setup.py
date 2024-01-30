from setuptools import setup


setup(name='pointscat',
      install_requires=['numpy', 'scipy', 'celer', 'jax', 'jaxopt'],
      description="Localization of point scatterers",
      packages=['pointscat']
      )

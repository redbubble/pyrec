from setuptools import setup

setup(name='pyrec',
      version='v0.1',
      description='Library to train and serve recommendation models in Python',
      url='https://github.com/redbubble/pyrec',
      author='Redbubble',
      author_email='info@redbubble.com',
      license='MIT',
      packages=['pyrec', 'pyrec.implicit'],
      install_requires=[
          'numpy',
          'scipy',
          'implicit>=0.5.0',
          'annoy>=1.17.1',
          'hnswlib>=0.6.2',
      ],
      zip_safe=False,
      )

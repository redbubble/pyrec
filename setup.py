from setuptools import setup

setup(name='redbubble',
      version='v0.1',
      description='Library to train and serve recommendation models in Python',
      url='https://github.com/redbubble/pyrec',
      author='Redbubble',
      author_email='info@redbubble.com',
      license='MIT',
      packages=['pyrec'],
      install_requires=[
          'numpy',
          'scipy',
          #     'implicit',
          #     'annoy',
          #     'nmslib',
      ],
      zip_safe=False,
      )

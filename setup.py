from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='gym_uav',
      version='0.0.1',
      install_requires=['gym', 'numpy']#And any other dependencies required
)

setup(name='drone_rl',
      version='0.0.1',
      packages=find_packages(),
      install_requires=required#And any other dependencies required
)
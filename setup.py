from setuptools import setup
from setuptools import find_packages

import re
import os
import sys
from pkg_resources import parse_version

# Dependencies of PILCO
requirements = [
    'pilco',
    'seaborn'
]

packages = find_packages('.')
setup(name='safe_pilco_experiments',
      version='0.1',
      author="Kyriakos Polymenakos",
      author_email="kpol@robots.ox.ac.uk",
      url="https://github.com/kyr-pol/SafePILCO_Tool-Reproducibility",
      packages=packages,
      install_requires=requirements,
      )

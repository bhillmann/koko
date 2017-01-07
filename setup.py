from setuptools import setup, find_packages
from glob import glob
import os

__author__ = "Benjamin Hillmann"
__copyright__ = "Copyright (c) 2016--, %s" % __author__
__credits__ = ["Benjamin Hillmann"]
__email__ = "hillmannben@gmail.com"
__license__ = "MIT"
__maintainer__ = "Benjamin Hillmann"
__version__ = "0.0.1-dev"

long_description = ''

setup(
    name='koko',
    version=__version__,
    packages=find_packages(),
    url='',
    license=__license__,
    author=__author__,
    author_email=__email__,
    description='',
    long_description=long_description,
    # scripts=glob(os.path.join('koko', 'scripts', '*py')),
    keywords='',
    install_requires=[],
    entry_points={
        'console_scripts': []
    },
)

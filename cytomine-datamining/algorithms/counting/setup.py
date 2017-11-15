# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='CellCounting',
    version='0.1',
    author='Ulysse Rubens',
    author_email='urubens@uliege.be',
    packages=['cell_counting', 'cell_counting.validation'],
    install_requires=['numpy', 'scikit-learn', 'scipy', 'keras', 'shapely', 'joblib']
)

#!/usr/bin/env python
# coding: utf-8
from setuptools import setup


setup(
    name='elm_estimator',
    version='0.0.1',
    description='An Extreme Learning Machine implementaion written in Python.',
    author='nel215',
    author_email='otomo.yuhei@gmail.com',
    url='https://github.com/nel215/elm_estimator',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['elm_estimator'],
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    keywords=['machine learning'],
)

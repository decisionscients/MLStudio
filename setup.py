#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as monitoring_file:
    monitoring = monitoring_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="John James",
    author_email='jjames@decisionscients.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Machine Learning Algorithms",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + monitoring,
    include_package_data=True,
    keywords='MLStudio',
    name='MLStudio',
    packages=find_packages(include=['mlstudio', 'mlstudio.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/decisionscients/MLStudio',
    version='0.1.15',
    zip_safe=False,
    package_data={"":['demo/data/Ames/*.csv']}
)

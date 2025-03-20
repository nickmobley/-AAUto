#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# Parse requirements.txt to get the list of dependencies
def parse_requirements(filename):
    """Parse a requirements file, removing comments and blank lines."""
    requirements = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines, comments, and section headers
            if not line or line.startswith('#') or line.startswith('--'):
                continue
            # Extract package name and version, ignoring duplicates
            package = line.split('==')[0].split('>=')[0].strip()
            if package not in [req.split('==')[0].split('>=')[0].strip() for req in requirements]:
                requirements.append(line)
    return requirements

# Get long description from README.md
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = """
    AAUto - Automated Trading and Investment System
    
    A sophisticated automated trading and investment system that integrates
    multiple strategies including trading, investments, and freelancing.
    It leverages market data from Alpha Vantage API and includes risk management,
    technical analysis, and machine learning components.
    """

# Get version from a version file or set manually
version = '0.1.0'

# Separate requirements by category
install_requires = parse_requirements('requirements.txt')
tests_require = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'pytest-mock>=3.11.1',
]
dev_requires = [
    'black>=23.7.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'mypy>=1.4.1',
    'bandit>=1.7.5',
    'pre-commit>=3.3.3',
]

setup(
    name='aauto',
    version=version,
    description='Automated Trading and Investment System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AAUto Team',
    author_email='info@aauto.example.com',
    url='https://github.com/yourusername/AAUto',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'test': tests_require,
        'all': dev_requires + tests_require,
    },
    entry_points={
        'console_scripts': [
            'aauto=aauto.main:main',
            'aauto-backtest=aauto.tools.backtest:main',
            'aauto-metrics=aauto.tools.metrics:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='trading, investment, machine learning, finance, algorithmic trading',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/AAUto/issues',
        'Source': 'https://github.com/yourusername/AAUto',
        'Documentation': 'https://aauto.readthedocs.io',
    },
    zip_safe=False,
)


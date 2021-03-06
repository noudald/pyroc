"""PyROC - A Python library for computing ROC curves."""

import setuptools

from pyroc import __version__

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name='PyROC',
    version=__version__,
    author='Noud Aldenhoven',
    author_email='noud.aldenhoven@gmail.com',
    description='A Python library for generating ROC curves.',
    long_description=LONG_DESCRIPTION,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent'
    ],
)

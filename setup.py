import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='PyROC',
    version='0.0.1',
    author='Noud Aldenhoven',
    author_email='noud.aldenhoven@gmail.com',
    description='A Python library for generating ROC curves.',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent'
    ],
)

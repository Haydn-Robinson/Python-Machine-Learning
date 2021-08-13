from setuptools import setup, find_packages

setup(
    name='machinelearningpython',
    version='1.0.0.dev0',
    author='Haydn Robinson',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    install_requires=['numpy',
                      'pandas',
                      'matplotlib'],
    extras_require={'test':['pytest']}
    )

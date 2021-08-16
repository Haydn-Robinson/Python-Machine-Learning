from setuptools import setup, find_packages

setup(
    name='hr-pyml',
    version='1.0.0.dev0',
    author='Haydn Robinson',
    description="A machine learning library developed for educational purposes.",
    url='https://github.com/Haydn-Robinson/Python-Machine-Learning',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    install_requires=['numpy',
                      'pandas',
                      'matplotlib'],
    extras_require={'test':['pytest']}
    )

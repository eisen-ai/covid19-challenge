from setuptools import setup, find_packages


VERSION = '0.0.1'

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')


setup(
    name='covid-challenge',
    version=VERSION,
    description='Covid 19 challenge: https://www.covid19challenge.eu',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'eisen_covid = covid_challenge.utils:cli'
        ],
    },
)

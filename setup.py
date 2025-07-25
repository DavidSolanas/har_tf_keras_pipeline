from setuptools import setup, find_packages

setup(
    name='har_pipeline',
    version='0.1.0',
    packages=find_packages(include=['har_pipeline', 'har_pipeline.*']),
    install_requires=[
        # List the required packages
    ],
    python_requires='>=3.10',
)

from setuptools import setup, find_packages

setup(
    name="SDMB-Sampler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
    ],
    author="Your Name",
    description="SDMB-Sampler: Density-based minority class balancer",
    url="https://github.com/yourusername/SDMB-Sampler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
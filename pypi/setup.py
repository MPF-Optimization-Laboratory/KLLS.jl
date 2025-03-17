from setuptools import setup, find_packages

setup(
    name="DualPerspective",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "juliacall>=0.9.7",
        "numpy>=1.20.0",
    ],
    author="Michael P. Friedlander",
    author_email="michael.friedlander@ubc.ca",
    description="Python interface for DualPerspective.jl - A Julia package for solving large-scale KL divergence problems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MPF-Optimization-Laboratory/DualPerspective.jl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 
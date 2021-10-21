from setuptools import setup, find_packages

setup(
    name="teaser",
    version="0.1.0",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=[
        "numpy==1.19.1",
        "scipy==1.5.2",
        "pandas==1.1.0",
        "tqdm==4.46.0",
        "matplotlib==3.4.3",
        "sklearn==0.0",
        "numba==0.54.1",
        "implicit==0.4.8",
    ],
    entry_points={},
)

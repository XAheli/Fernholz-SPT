from setuptools import setup, find_packages

setup(
    name="fernholz_spt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=0.24.0',
        'cvxpy>=1.1.0'
    ],
    author="Aheli Poddar",
    author_email="ahelipoddar2003@gmail.com",
    description="Implementation of Fernholz's Stochastic Portfolio Theory",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fernholz_spt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

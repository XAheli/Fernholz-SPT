from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="fernholz_spt",
    version="0.1.1", # Incremented version
    author="Aheli Poddar",
    author_email="ahelipoddar2003@gmail.com",
    description="A Python Implementation of Fernholz's Stochastic Portfolio Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xaheli/fernholz-spt",
    packages=find_packages(exclude=['tests*', 'examples*']),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=1.0.0',
        'cvxpy>=1.2.0',
        'yfinance>=0.2.0',
        'seaborn>=0.11.2',
        'statsmodels>=0.13.0'
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=22.1.0",
            "flake8>=4.0.1",
            "mypy>=0.931",
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0", # For Readthedocs theme
            "ipywidgets>=7.6.5", # For notebooks
            "plotly>=5.5.0", # For interactive plots
        ],
        "performance": [
            "numba>=0.55.0",
            "joblib>=1.1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.8',
    keywords='stochastic portfolio theory, fernholz, quantitative finance, portfolio optimization, finance',
)
from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="option-portfolio-skew-t",
    version="1.0.0",
    author="Nicholas Rebello",
    author_email="nicholas.rebello@gmail.ca",
    description="Sharpe Ratio and Return-VaR Ratio Maximization for Option Portfolios with Skew-Elliptical t Returns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nickr145/option-portfolio-skew-t",
    packages=find_packages(),
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
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "data": ["yfinance>=0.1.70"],
        "notebooks": ["jupyter>=1.0.0", "ipython>=7.0.0"],
    },
    keywords="quantitative-finance option-pricing portfolio-optimization skew-t",
    project_urls={
        "Bug Reports": "https://github.com/nickr145/option-portfolio-skew-t/issues",
        "Source": "https://github.com/nickr145/option-portfolio-skew-t",
        "Documentation": "https://github.com/nickr145/option-portfolio-skew-t#readme",
    },
)

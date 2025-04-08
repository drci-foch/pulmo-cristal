"""Setup script for the pulmo-cristal package."""

from setuptools import setup, find_packages

setup(
    name="pulmo-cristal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyPDF2>=3.0.0",
        "camelot-py[cv]>=0.10.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],

    },
    entry_points={
        'console_scripts': [
            'pulmo-cristal=pulmo_cristal.cli:main',
        ],
    },
)
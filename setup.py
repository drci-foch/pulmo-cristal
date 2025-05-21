# setup.py
from setuptools import setup, find_packages

setup(
    name="pulmo-cristal",
    version="0.1.0",
    description="Extract data from donor PDF documents for pulmonary transplantation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "PyPDF2>=3.0.0",
        "camelot-py>=1.0.0",
        "PyPDF2>=3.0.0",
        "opencv-python-headless>=4.5.0",  # Required for Camelot
        "ghostscript>=0.7",  # Required for Camelot
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
        "console_scripts": [
            "pulmo-cristal=pulmo_cristal.cli:main",
        ],
    },
)

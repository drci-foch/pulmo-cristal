[![PyPI](https://img.shields.io/pypi/v/pulmo-cristal?color=blue&style=flat-square)](https://pypi.org/project/pulmo-cristal/)

# pulmo-cristal

A Python package for extracting and structuring donor data from PDF documents for pulmonary transplantation research.

## Overview

pulmo-cristal is a specialized tool for medical researchers working with organ donation documents. It extracts structured data from donor PDF files (known as "Cristal" documents), enabling researchers to analyze key patient information for pulmonary transplantation studies.

Key features:
- Extract donor information including demographics, clinical parameters, and HLA typing
- Process PDF documents with table detection and text extraction
- Structure data into standardized formats (JSON, CSV)
- Support batch processing of multiple documents
- Provide a command-line interface for easy operation

## Installation

### Prerequisites

pulmo-cristal requires some system dependencies:

**On Ubuntu/Debian:**
```bash
sudo apt-get install ghostscript python3-opencv
```

**On macOS:**
```bash
brew install ghostscript
```

**On Windows:**
- Install Ghostscript from [the official website](https://www.ghostscript.com/download/gsdnld.html)
- Add it to your PATH environment variable

### Installing the package

```bash
# Install directly from GitHub
pip install git+https://github.com/drci-foch/pulmo-cristal.git
```

For development installation:
```bash
# Clone the repository
git clone https://github.com/drci-foch/pulmo-cristal.git
cd pulmo-cristal

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies first (important for Camelot)
pip install camelot-py[cv] opencv-python-headless ghostscript

# Install in development mode
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line interface for common operations:

```bash
# Show help and available commands
pulmo-cristal --help

# List PDF files in a directory
pulmo-cristal list --input /path/to/pdfs

# Extract data from PDF files
pulmo-cristal extract --input /path/to/pdfs --output /path/to/output --format json

# Process recursively through subdirectories
pulmo-cristal extract --input /path/to/pdfs --recursive

# Validate extracted data
pulmo-cristal validate --input /path/to/output/donneurs_data.json

# Convert JSON to CSV
pulmo-cristal convert --input /path/to/output/donneurs_data.json --output /path/to/output/donneurs_data.csv
```

### Python API

For more advanced scenarios, you can use pulmo-cristal directly in your Python code:

```python
from pulmo_cristal.extractors import DonorPDFExtractor, HLAExtractor
from pulmo_cristal.exporters import DonorCSVExporter, DonorJSONExporter
from pulmo_cristal.utils import find_pdf_files

# Find PDF files
pdf_files = find_pdf_files("/path/to/pdfs", recursive=True)

# Extract data from a single PDF
pdf_path = pdf_files[0]
donor_extractor = DonorPDFExtractor()
hla_extractor = HLAExtractor()

# Extract donor data
donor_data = donor_extractor.extract_donor_data(pdf_path)

# Extract HLA data
hla_data, status = hla_extractor.extract_hla_data(pdf_path)
donor_data["informations_donneur"]["hla"] = hla_data
donor_data["informations_donneur"]["hla_extraction_status"] = status

# Export to JSON
json_exporter = DonorJSONExporter()
json_exporter.export_json([donor_data], "/path/to/output/data.json")

# Export to CSV
csv_exporter = DonorCSVExporter()
csv_exporter.export_csv([donor_data], "/path/to/output/data.csv")
```

## Troubleshooting

### Common Issues

1. **HLA extraction fails**:
   - Make sure Ghostscript and OpenCV are properly installed
   - Check the PDF format and structure
   - Try using the debug mode: `DonorPDFExtractor(debug=True)`

2. **Import errors with Camelot**:
   - Install dependencies separately: `pip install camelot-py[cv] opencv-python-headless ghostscript`
   - Check system dependencies (Ghostscript, etc.)

3. **Processing takes too long**:
   - Use batch processing with smaller batch sizes
   - Process files in parallel with multiple workers

### Testing Camelot Installation

Use this script to test if Camelot is correctly installed:

```python
# test_camelot.py
import sys
import traceback

print("Testing Camelot installation...")

try:
    import camelot
    print(f"Camelot version: {camelot.__version__}")
    
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    
    import ghostscript
    print("Ghostscript is available")
    
    # Test with a PDF if provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Testing with: {pdf_path}")
        
        tables = camelot.read_pdf(
            pdf_path,
            flavor="stream",
            pages="1"
        )
        
        print(f"Extracted {len(tables)} tables")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

## Project Structure

```
pulmo-cristal/
├── pulmo_cristal/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── extractors/         # PDF extraction modules
│   ├── exporters/          # Data export modules
│   ├── models/             # Data models
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── setup.py                # Package setup
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

See the LICENSE file for details.

## Acknowledgments

- This project was developed for pulmonary transplantation research at Hôpital Foch
- Special thanks to the DRCI (Département de la Recherche Clinique et de l'Innovation)
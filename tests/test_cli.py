# tests/test_cli_functional.py
import subprocess
import os
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Create a temporary test directory with sample files."""
    test_dir = Path("tests/data/cli_test")
    test_dir.mkdir(exist_ok=True, parents=True)
    # Add sample PDFs to this directory
    return test_dir

def test_cli_list_command(test_data_dir):
    """Test the list command."""
    result = subprocess.run(
        ["python", "-m", "pulmo_cristal.cli", "list", "--input", str(test_data_dir)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Found " in result.stdout

def test_cli_extract_command(test_data_dir):
    """Test the extract command."""
    output_dir = Path("tests/data/cli_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    result = subprocess.run(
        [
            "python", "-m", "pulmo_cristal.cli", 
            "extract", 
            "--input", str(test_data_dir),
            "--output", str(output_dir),
            "--format", "json"
        ],
        capture_output=True,
        text=True
    )
    
    # Check results
    assert result.returncode == 0
    assert any(output_dir.glob("*.json"))
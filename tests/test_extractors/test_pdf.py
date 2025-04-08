import pytest
from unittest.mock import patch, mock_open
from pulmo_cristal.extractors.pdf import PDFExtractor

def test_extract_from_pdf_with_mock():
    """Test PDF extraction using a mock PDF file."""
    # Mock the PDF reader
    with patch('PyPDF2.PdfReader') as mock_reader:
        # Configure the mock
        mock_page = type('MockPage', (), {'extract_text': lambda: 'Test text content'})
        mock_reader.return_value.pages = [mock_page, mock_page]
        
        # Create the extractor
        extractor = PDFExtractor()
        
        # Use mock_open to simulate a file
        with patch('builtins.open', mock_open()):
            text, pages = extractor.extract_from_pdf("fake.pdf")
            
            # Assertions
            assert len(pages) == 2
            assert "Test text content" in text
            assert mock_reader.called
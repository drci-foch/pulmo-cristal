"""
Extractors package for pulmo-cristal.

This package provides various extractors for processing donor PDF documents
and extracting structured data for pulmonary transplantation.
"""

from .base import BaseExtractor
from .pdf import PDFExtractor, DonorPDFExtractor
from .hla import HLAExtractor, HLAData
from .patterns import get_pattern_group, create_custom_pattern
from .gds import ImprovedGazDuSangExtractor, GDSData
from .hemodynamic import ImprovedHemodynamicExtractor

__all__ = [
    # Base classes
    "BaseExtractor",
    # PDF extractors
    "PDFExtractor",
    "DonorPDFExtractor",
    # HLA extractor
    "HLAExtractor",
    "HLAData",
    # Pattern utilities
    "get_pattern_group",
    "create_custom_pattern",
    # Specialized extractors - NOUVEAUX
    "ImprovedGazDuSangExtractor",
    "GDSData",
    "ImprovedHemodynamicExtractor",
]

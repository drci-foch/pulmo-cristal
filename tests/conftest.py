import pytest
from datetime import datetime

@pytest.fixture
def mock_donor_data():
    """Create a mock donor data dictionary for testing."""
    return {
        "informations_donneur": {
            "num_cristal": "12345",
            "type_donneur": "mort encéphalique",
            "age": "45",
            "sexe": "M",
            "taille": "175",
            "poids": "70",
            "hla": {
                "A1": "1", "A2": "2", "B1": "8", "B2": "44",
                "C1": "5", "C2": "7", "DR1": "11", "DR2": "13"
            }
        },
        "serologies": {
            "anti_hiv": "NEGATIF",
            "anti_hcv": "NEGATIF"
        },
        "fichier_source": "test_file.pdf",
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
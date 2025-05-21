"""Tests for the models.donneur module."""

from pulmo_cristal.models.donneurs import SexeType, DonneurType, BooleanValue, Donneur


def test_boolean_value_from_string():
    """Test BooleanValue.from_string method."""
    assert BooleanValue.from_string("OUI") == BooleanValue.OUI
    assert BooleanValue.from_string("NON") == BooleanValue.NON
    assert BooleanValue.from_string("") == BooleanValue.INCONNU


def test_sexe_type_from_string():
    """Test SexeType.from_string method."""
    assert SexeType.from_string("M") == SexeType.HOMME
    assert SexeType.from_string("F") == SexeType.FEMME
    assert SexeType.from_string("") == SexeType.INCONNU


def test_donneur_type_from_string():
    """Test DonneurType.from_string method."""
    assert DonneurType.from_string("mort encéphalique") == DonneurType.DBD
    assert DonneurType.from_string("arrêt circulatoire") == DonneurType.DCD
    assert DonneurType.from_string("") == DonneurType.INCONNU


def test_donneur_creation():
    """Test creating a Donneur object."""
    d = Donneur(id="12345", age=45, sexe=SexeType.HOMME)
    assert d.id == "12345"
    assert d.age == 45
    assert d.sexe == SexeType.HOMME
    assert d.type_donneur == DonneurType.INCONNU  # Default value

"""
Donneur Models Module for pulmo-cristal package.

This module defines data models for representing donor (donneur) data
using dataclasses for type safety and structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum, auto


class SexeType(Enum):
    """Enumeration of possible donor sex types."""

    HOMME = "M"
    FEMME = "F"
    INCONNU = "I"

    @classmethod
    def from_string(cls, value: str) -> "SexeType":
        """Convert a string representation to enum value."""
        value = value.upper().strip() if value else "I"

        if value == "M" or value == "H":
            return cls.HOMME
        elif value == "F":
            return cls.FEMME
        else:
            return cls.INCONNU


class DonneurType(Enum):
    """Enumeration of possible donor types."""

    DBD = auto()  # Donation after Brain Death
    DCD = auto()  # Donation after Circulatory Death
    INCONNU = auto()

    @classmethod
    def from_string(cls, value: str) -> "DonneurType":
        """Convert a string representation to enum value."""
        value = value.lower().strip() if value else ""

        if "mort encéphalique" in value or "me " in value or "dbd" in value:
            return cls.DBD
        elif "arrêt circulatoire" in value or "maastricht" in value or "dcd" in value:
            return cls.DCD
        else:
            return cls.INCONNU


class BooleanValue(Enum):
    """Enumeration for representing boolean values with uncertain states."""

    OUI = "OUI"
    NON = "NON"
    INCONNU = "INCONNU"

    @classmethod
    def from_string(cls, value: str) -> "BooleanValue":
        """Convert a string representation to enum value."""
        if not value:
            return cls.INCONNU

        value = value.upper().strip()

        if value in ("OUI", "YES", "Y", "O", "1", "POSITIF", "POSITIVE", "+"):
            return cls.OUI
        elif value in ("NON", "NO", "N", "0", "NEGATIF", "NEGATIVE", "-"):
            return cls.NON
        else:
            return cls.INCONNU


@dataclass
class HLAData:
    """HLA (Human Leukocyte Antigen) typing data."""

    A1: str = ""
    A2: str = ""
    B1: str = ""
    B2: str = ""
    C1: str = ""
    C2: str = ""
    DR1: str = ""
    DR2: str = ""
    DQB1: str = ""
    DQB2: str = ""
    DP1: str = ""
    DP2: str = ""
    extraction_status: str = "INCONNU"

    def is_valid(self) -> bool:
        """Check if the HLA data has meaningful values."""
        required_fields = ["A1", "A2", "B1", "B2", "DR1", "DR2"]
        return all(
            getattr(self, field) not in ("", "À AJOUTER", "INCONNU")
            for field in required_fields
        )


@dataclass
class SerologiesData:
    """Serological test results."""

    antigene_p24: Optional[BooleanValue] = None
    combine_hiv: Optional[BooleanValue] = None
    dgv_vih: Optional[BooleanValue] = None
    dgv_vhc: Optional[BooleanValue] = None
    dgv_vhb: Optional[BooleanValue] = None
    anti_htlv: Optional[BooleanValue] = None
    anti_hcv: Optional[BooleanValue] = None
    antigene_hbs: Optional[BooleanValue] = None
    anti_hbc: Optional[BooleanValue] = None
    anti_hbs: Optional[BooleanValue] = None
    dgv_vhe: Optional[BooleanValue] = None
    serologie_anguillulose: Optional[BooleanValue] = None
    syphilis_tpha: Optional[BooleanValue] = None
    anti_cmv: Optional[BooleanValue] = None
    anti_ebv: Optional[BooleanValue] = None
    anti_toxoplasmose: Optional[BooleanValue] = None
    anti_hhv8: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "SerologiesData":
        """Create a SerologiesData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class MorphologieData:
    """Morphological measurements data."""

    perimetre_bi_mamelonnaire: Optional[float] = None
    hauteur_sternale: Optional[float] = None
    perimetre_ombilical: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "MorphologieData":
        """Create a MorphologieData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name) and field_value:
                try:
                    # Try to convert to float
                    numeric_value = float(field_value.replace(",", "."))
                    setattr(result, field_name, numeric_value)
                except (ValueError, TypeError):
                    # Keep as None if conversion fails
                    pass

        return result


@dataclass
class HabitusData:
    """Lifestyle/habits data."""

    alcoolisme: Optional[BooleanValue] = None
    tabagisme: Optional[BooleanValue] = None
    toxicomanie: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "HabitusData":
        """Create a HabitusData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class AntecedentsData:
    """Medical history data."""

    traitement: Optional[BooleanValue] = None
    hta: Optional[BooleanValue] = None
    diabete: Optional[BooleanValue] = None
    maladie_broncho_pulmonaire: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "AntecedentsData":
        """Create an AntecedentsData object from a dictionary."""
        result = cls()

        # Handle potential naming inconsistencies
        mapping = {
            "maladie_broncho_pulmonanire": "maladie_broncho_pulmonaire",
        }

        for field_name, field_value in data.items():
            # Map alternative field names
            if field_name in mapping:
                field_name = mapping[field_name]

            if hasattr(result, field_name):
                setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class BilanInfectieuxData:
    """Infectious disease assessment data."""

    antibiotherapie: Optional[BooleanValue] = None
    si_oui_preciser: str = ""
    diag_covid19: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BilanInfectieuxData":
        """Create a BilanInfectieuxData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                if field_name == "si_oui_preciser":
                    setattr(result, field_name, str(field_value).strip())
                else:
                    setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class BilanHemodynamiqueData:
    """Hemodynamic assessment data."""

    arret_cardiaque_recup: Optional[BooleanValue] = None
    concentre_globulaire: Optional[int] = None
    plasma_frais_congele: Optional[int] = None
    concentre_plaquettaire: Optional[int] = None
    albumine: Optional[BooleanValue] = None
    autres_medicaments: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BilanHemodynamiqueData":
        """Create a BilanHemodynamiqueData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                if field_name in [
                    "concentre_globulaire",
                    "plasma_frais_congele",
                    "concentre_plaquettaire",
                ]:
                    try:
                        setattr(result, field_name, int(field_value))
                    except (ValueError, TypeError):
                        pass
                elif field_name == "autres_medicaments":
                    setattr(result, field_name, str(field_value).strip())
                else:
                    setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class EvolutionHemodynamiqueData:
    """Hemodynamic evolution data."""

    dopamine: Optional[float] = None
    dobutamine: Optional[float] = None
    adrenaline: Optional[float] = None
    noradrenaline: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EvolutionHemodynamiqueData":
        """Create an EvolutionHemodynamiqueData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name) and field_value:
                try:
                    # Try to convert to float
                    numeric_value = float(str(field_value).replace(",", "."))
                    setattr(result, field_name, numeric_value)
                except (ValueError, TypeError):
                    # Keep as None if conversion fails
                    pass

        return result


@dataclass
class BilanPulmonaireData:
    """Pulmonary assessment data."""

    traumatise_broncho_pulmonaire_actuel: Optional[BooleanValue] = None
    lesion_pleurale_traumatique_actuelle: Optional[BooleanValue] = None
    radiographie_thoraco_pulmonaire: str = ""
    aspirations_tracheo_bronchiques: str = ""
    prelevement_bacteriologique: Optional[BooleanValue] = None
    fibroscopie_bronchique: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BilanPulmonaireData":
        """Create a BilanPulmonaireData object from a dictionary."""
        result = cls()

        # Handle potential naming inconsistencies
        mapping = {
            "radiographie_thoraco-pulmonaire": "radiographie_thoraco_pulmonaire",
            "aspirations trachéo-bronchiques": "aspirations_tracheo_bronchiques",
        }

        for field_name, field_value in data.items():
            # Map alternative field names
            if field_name in mapping:
                field_name = mapping[field_name]

            if hasattr(result, field_name):
                if field_name in [
                    "radiographie_thoraco_pulmonaire",
                    "aspirations_tracheo_bronchiques",
                ]:
                    setattr(result, field_name, str(field_value).strip())
                else:
                    setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class ParametresRespiratoiresData:
    """Respiratory parameters data."""

    pH: Optional[float] = None
    PaCO2: Optional[float] = None
    PaO2: Optional[float] = None
    CO3H: Optional[float] = None
    SaO2: Optional[float] = None
    PEEP: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ParametresRespiratoiresData":
        """Create a ParametresRespiratoiresData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name) and field_value:
                try:
                    # Try to convert to float
                    numeric_value = float(str(field_value).replace(",", "."))
                    setattr(result, field_name, numeric_value)
                except (ValueError, TypeError):
                    # Keep as None if conversion fails
                    pass

        return result


@dataclass
class BilanCardiaqueData:
    """Cardiac assessment data."""

    fraction_d_ejection: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BilanCardiaqueData":
        """Create a BilanCardiaqueData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                setattr(result, field_name, str(field_value).strip())

        return result


@dataclass
class ThoraxData:
    """Thorax assessment data."""

    epanchement_gazeux_droit: Optional[BooleanValue] = None
    epanchement_gazeux_gauche: Optional[BooleanValue] = None
    epanchement_liquidien_droit: Optional[BooleanValue] = None
    epanchement_liquidien_gauche: Optional[BooleanValue] = None
    atelectasie_droit: Optional[BooleanValue] = None
    atelectasie_gauche: Optional[BooleanValue] = None
    contusion_pulmonaire_droit: Optional[BooleanValue] = None
    contusion_pulmonaire_gauche: Optional[BooleanValue] = None
    infiltrat_droit: Optional[BooleanValue] = None
    infiltrat_gauche: Optional[BooleanValue] = None
    images_compatibles_avec_inhalation_droit: Optional[BooleanValue] = None
    images_compatibles_avec_inhalation_gauche: Optional[BooleanValue] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ThoraxData":
        """Create a ThoraxData object from a dictionary."""
        result = cls()

        for field_name, field_value in data.items():
            if hasattr(result, field_name):
                setattr(result, field_name, BooleanValue.from_string(field_value))

        return result


@dataclass
class Donneur:
    """
    Main donor (donneur) data model that aggregates all information.

    This class represents a complete donor record with all associated data.
    """

    # Basic donor information
    id: str = ""  # Typically the CRISTAL number
    type_donneur: DonneurType = DonneurType.INCONNU
    age: Optional[int] = None
    sexe: SexeType = SexeType.INCONNU
    groupe_sanguin: str = ""
    date_naissance: Optional[date] = None
    taille: Optional[int] = None  # in cm
    poids: Optional[float] = None  # in kg

    # Important dates
    date_creation: Optional[datetime] = None
    date_entree_bloc: Optional[datetime] = None
    date_clampage: Optional[datetime] = None

    # Medical information
    etiologie: str = ""  # Cause of death
    duree_ventilation: str = ""

    # Structured medical data
    hla: Optional[HLAData] = None
    serologies: Optional[SerologiesData] = None
    morphologie: Optional[MorphologieData] = None
    habitus: Optional[HabitusData] = None
    antecedents: Optional[AntecedentsData] = None
    bilan_infectieux: Optional[BilanInfectieuxData] = None
    bilan_hemodynamique: Optional[BilanHemodynamiqueData] = None
    evolution_hemodynamique: Optional[EvolutionHemodynamiqueData] = None
    bilan_pulmonaire: Optional[BilanPulmonaireData] = None
    parametres_respiratoires: Optional[ParametresRespiratoiresData] = None
    bilan_cardiaque_morphologique: Optional[BilanCardiaqueData] = None
    thorax: Optional[ThoraxData] = None

    # File information
    fichier_source: str = ""
    chemin_relatif: str = ""
    date_extraction: Optional[datetime] = None

    # Validation status
    validation_errors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Donneur":
        """
        Create a Donneur object from a raw data dictionary.

        Args:
            data: Dictionary containing extracted donor data

        Returns:
            Initialized Donneur object
        """
        donneur = cls()

        # Set file information
        donneur.fichier_source = data.get("fichier_source", "")
        donneur.chemin_relatif = data.get("chemin_relatif", "")

        # Extract date_extraction
        if "date_extraction" in data:
            try:
                donneur.date_extraction = datetime.fromisoformat(
                    data["date_extraction"]
                )
            except (ValueError, TypeError):
                try:
                    donneur.date_extraction = datetime.strptime(
                        data["date_extraction"], "%Y-%m-%d %H:%M:%S"
                    )
                except (ValueError, TypeError):
                    pass

        # Process donor information
        info_donneur = data.get("informations_donneur", {})
        if info_donneur:
            # Basic identifiers
            donneur.id = info_donneur.get("num_cristal", "")
            donneur.type_donneur = DonneurType.from_string(
                info_donneur.get("type_donneur", "")
            )

            # Demographics
            if "age" in info_donneur:
                try:
                    donneur.age = int(info_donneur["age"])
                except (ValueError, TypeError):
                    pass

            donneur.sexe = SexeType.from_string(info_donneur.get("sexe", ""))
            donneur.groupe_sanguin = info_donneur.get("groupe_sanguin", "")

            # Parse date of birth
            if "date_naissance" in info_donneur:
                try:
                    donneur.date_naissance = datetime.strptime(
                        info_donneur["date_naissance"], "%d/%m/%Y"
                    ).date()
                except (ValueError, TypeError):
                    pass

            # Physical measurements
            if "taille" in info_donneur:
                try:
                    donneur.taille = int(info_donneur["taille"])
                except (ValueError, TypeError):
                    pass

            if "poids" in info_donneur:
                try:
                    donneur.poids = float(str(info_donneur["poids"]).replace(",", "."))
                except (ValueError, TypeError):
                    pass

            # Important dates
            for date_field in ["date_creation", "date_entree_bloc", "date_clampage"]:
                if date_field in info_donneur:
                    try:
                        setattr(
                            donneur,
                            date_field,
                            datetime.strptime(info_donneur[date_field], "%d/%m/%Y"),
                        )
                    except (ValueError, TypeError):
                        pass

            # Medical information
            donneur.etiologie = info_donneur.get("etiologie", "")
            donneur.duree_ventilation = info_donneur.get("duree_ventilation", "")

            # HLA data
            if "hla" in info_donneur:
                hla_data = HLAData()
                for key, value in info_donneur["hla"].items():
                    if hasattr(hla_data, key):
                        setattr(hla_data, key, value)

                if "hla_extraction_status" in info_donneur:
                    hla_data.extraction_status = info_donneur["hla_extraction_status"]

                donneur.hla = hla_data

        # Process specialized data sections
        if "serologies" in data:
            donneur.serologies = SerologiesData.from_dict(data["serologies"])

        if "morphologie" in data:
            donneur.morphologie = MorphologieData.from_dict(data["morphologie"])

        if "habitus" in data:
            donneur.habitus = HabitusData.from_dict(data["habitus"])

        if "antecedents" in data:
            donneur.antecedents = AntecedentsData.from_dict(data["antecedents"])

        if "bilan_infectieux" in data:
            donneur.bilan_infectieux = BilanInfectieuxData.from_dict(
                data["bilan_infectieux"]
            )

        if "bilan_hemodynamique" in data:
            donneur.bilan_hemodynamique = BilanHemodynamiqueData.from_dict(
                data["bilan_hemodynamique"]
            )

        if "evolution_hemodynamique" in data:
            donneur.evolution_hemodynamique = EvolutionHemodynamiqueData.from_dict(
                data["evolution_hemodynamique"]
            )

        if "bilan_pulmonaire" in data:
            donneur.bilan_pulmonaire = BilanPulmonaireData.from_dict(
                data["bilan_pulmonaire"]
            )

        if "parametres_respiratoires" in data:
            donneur.parametres_respiratoires = ParametresRespiratoiresData.from_dict(
                data["parametres_respiratoires"]
            )

        if "bilan_cardiaque_morphologique" in data:
            donneur.bilan_cardiaque_morphologique = BilanCardiaqueData.from_dict(
                data["bilan_cardiaque_morphologique"]
            )

        if "thorax" in data:
            donneur.thorax = ThoraxData.from_dict(data["thorax"])

        # Validate the donor data
        donneur.validate()

        return donneur

    def validate(self) -> bool:
        """
        Validate the donor data.

        Returns:
            True if valid, False otherwise
        """
        self.validation_errors = []

        # Check required fields
        if not self.id:
            self.validation_errors.append("Numéro CRISTAL manquant")

        # Check age is in reasonable range
        if self.age is not None and (self.age < 0 or self.age > 120):
            self.validation_errors.append(f"Âge suspect: {self.age}")

        # Check height is in reasonable range
        if self.taille is not None and (self.taille < 50 or self.taille > 250):
            self.validation_errors.append(f"Taille suspecte: {self.taille} cm")

        # Check weight is in reasonable range
        if self.poids is not None and (self.poids < 20 or self.poids > 250):
            self.validation_errors.append(f"Poids suspect: {self.poids} kg")

        # Check dates make sense
        if self.date_naissance and self.date_clampage:
            if self.date_naissance > self.date_clampage.date():
                self.validation_errors.append(
                    "Date de naissance postérieure à la date de clampage"
                )

        if self.date_creation and self.date_clampage:
            if self.date_creation > self.date_clampage:
                self.validation_errors.append(
                    "Date de création postérieure à la date de clampage"
                )

        # Check file information
        if not self.fichier_source:
            self.validation_errors.append("Fichier source manquant")

        return len(self.validation_errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the donor object to a dictionary.

        Returns:
            Dictionary representation of the donor
        """
        result = {
            "id": self.id,
            "type_donneur": self.type_donneur.name if self.type_donneur else "INCONNU",
            "age": self.age,
            "sexe": self.sexe.name if self.sexe else "INCONNU",
            "groupe_sanguin": self.groupe_sanguin,
            "date_naissance": self.date_naissance.isoformat()
            if self.date_naissance
            else None,
            "taille": self.taille,
            "poids": self.poids,
            "date_creation": self.date_creation.isoformat()
            if self.date_creation
            else None,
            "date_entree_bloc": self.date_entree_bloc.isoformat()
            if self.date_entree_bloc
            else None,
            "date_clampage": self.date_clampage.isoformat()
            if self.date_clampage
            else None,
            "etiologie": self.etiologie,
            "duree_ventilation": self.duree_ventilation,
            "fichier_source": self.fichier_source,
            "chemin_relatif": self.chemin_relatif,
            "date_extraction": self.date_extraction.isoformat()
            if self.date_extraction
            else None,
            "validation_status": "VALID" if not self.validation_errors else "INVALID",
            "validation_errors": self.validation_errors,
        }

        # Add sub-objects if they exist
        if self.hla:
            result["hla"] = {k: v for k, v in self.hla.__dict__.items()}

        # Add all other data objects
        for field_name in [
            "serologies",
            "morphologie",
            "habitus",
            "antecedents",
            "bilan_infectieux",
            "bilan_hemodynamique",
            "evolution_hemodynamique",
            "bilan_pulmonaire",
            "parametres_respiratoires",
            "bilan_cardiaque_morphologique",
            "thorax",
        ]:
            obj = getattr(self, field_name)
            if obj:
                # Convert enums to strings
                obj_dict = {}
                for k, v in obj.__dict__.items():
                    if isinstance(v, Enum):
                        obj_dict[k] = v.name
                    else:
                        obj_dict[k] = v

                result[field_name] = obj_dict

        return result

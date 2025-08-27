"""
CSV Exporter Module for pulmo-cristal package.

This module provides functionality to export donor data to CSV format.
It handles data normalization, column definition, and file generation.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Local imports
try:
    from ..extractors.base import BaseExtractor
except ImportError:
    # If used standalone outside package structure
    BaseExtractor = object


class DonorCSVExporter(BaseExtractor):
    """
    Exporter for generating CSV files from donor data.

    This class handles the conversion of donor data dictionaries into CSV format,
    with properly defined columns, data normalization, and encoding settings.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        delimiter: str = ";",
        encoding: str = "utf-8-sig",
    ):
        """
        Initialize the CSV exporter.

        Args:
            logger: Optional logger instance
            delimiter: CSV delimiter character (default: semicolon)
            encoding: File encoding (default: utf-8-sig, which includes BOM for Excel)
        """
        super().__init__(logger=logger)
        self.delimiter = delimiter
        self.encoding = encoding

        # Define the column structure
        self.columns = self._define_columns()

    def _define_columns(self) -> List[Dict[str, str]]:
        """
        Define the columns and their mappings to the donor data structure.

        Returns:
            List of column definition dictionaries
        """
        return [
            # Basic donor information
            {
                "name": "num_cristal",
                "header": "Numéro CRISTAL",
                "path": ["informations_donneur", "num_cristal"],
            },
            {
                "name": "type_donneur",
                "header": "Type de donneur",
                "path": ["informations_donneur", "type_donneur"],
            },
            {"name": "age", "header": "Âge", "path": ["informations_donneur", "age"]},
            {
                "name": "sexe",
                "header": "Sexe",
                "path": ["informations_donneur", "sexe"],
            },
            {
                "name": "groupe_sanguin",
                "header": "Groupe sanguin",
                "path": ["informations_donneur", "groupe_sanguin"],
            },
            {
                "name": "date_naissance",
                "header": "Date de naissance",
                "path": ["informations_donneur", "date_naissance"],
            },
            {
                "name": "taille",
                "header": "Taille (cm)",
                "path": ["informations_donneur", "taille"],
            },
            {
                "name": "poids",
                "header": "Poids (kg)",
                "path": ["informations_donneur", "poids"],
            },
            {
                "name": "date_clampage",
                "header": "Date de clampage",
                "path": ["informations_donneur", "date_clampage"],
            },
            {
                "name": "heure_clampage",
                "header": "Heure de clampage",
                "path": ["informations_donneur", "heure_clampage"],
            },
            {
                "name": "etiologie",
                "header": "Étiologie (cause de décès)",
                "path": ["informations_donneur", "etiologie"],
            },
            {
                "name": "duree_ventilation",
                "header": "Durée de ventilation (jours)",
                "path": ["informations_donneur", "duree_ventilation"],
            },
            {
                "name": "commentaire",
                "header": "Commentaire",
                "path": ["informations_donneur", "commentaire"],
            },
            # HLA data
            {"name": "hla_A1", "header": "HLA A1", "path": ["informations_donneur", "hla", "A1"]},
            {"name": "hla_A2", "header": "HLA A2", "path": ["informations_donneur", "hla", "A2"]},
            {"name": "hla_B1", "header": "HLA B1", "path": ["informations_donneur", "hla", "B1"]},
            {"name": "hla_B2", "header": "HLA B2", "path": ["informations_donneur", "hla", "B2"]},
            {"name": "hla_C1", "header": "HLA C1", "path": ["informations_donneur", "hla", "C1"]},
            {"name": "hla_C2", "header": "HLA C2", "path": ["informations_donneur", "hla", "C2"]},
            {"name": "hla_DR1", "header": "HLA DR1", "path": ["informations_donneur", "hla", "DR1"]},
            {"name": "hla_DR2", "header": "HLA DR2", "path": ["informations_donneur", "hla", "DR2"]},
                    # NOUVELLES COLONNES HLA SPLITTÉES
            {"name": "hla_DQA1", "header": "HLA DQA1", "path": ["informations_donneur", "hla", "DQA1"]},
            {"name": "hla_DQA2", "header": "HLA DQA2", "path": ["informations_donneur", "hla", "DQA2"]},
            {"name": "hla_DQB1", "header": "HLA DQB1", "path": ["informations_donneur", "hla", "DQB1"]},
            {"name": "hla_DQB2", "header": "HLA DQB2", "path": ["informations_donneur", "hla", "DQB2"]},
            
            {"name": "hla_DP1", "header": "HLA DP1", "path": ["informations_donneur", "hla", "DP1"]},
            {"name": "hla_DP2", "header": "HLA DP2", "path": ["informations_donneur", "hla", "DP2"]},
            
            # {"name": "hla_extraction_status", "header": "Statut extraction HLA", "path": ["informations_donneur", "hla_extraction_status"]},
            # Serologies
            {
                "name": "antigene_p24",
                "header": "Antigène P24",
                "path": ["serologies", "antigene_p24"],
            },
            {
                "name": "combine_hiv",
                "header": "Combiné HIV",
                "path": ["serologies", "combine_hiv"],
            },
            {"name": "dgv_vih", "header": "DGV VIH", "path": ["serologies", "dgv_vih"]},
            {"name": "dgv_vhc", "header": "DGV VHC", "path": ["serologies", "dgv_vhc"]},
            {"name": "dgv_vhb", "header": "DGV VHB", "path": ["serologies", "dgv_vhb"]},
            {
                "name": "anti_htlv",
                "header": "Anti-HTLV",
                "path": ["serologies", "anti_htlv"],
            },
            {
                "name": "anti_hcv",
                "header": "Anti-HCV",
                "path": ["serologies", "anti_hcv"],
            },
            {
                "name": "antigene_hbs",
                "header": "Antigène HBs",
                "path": ["serologies", "antigene_hbs"],
            },
            {
                "name": "anti_hbc",
                "header": "Anti-HBc",
                "path": ["serologies", "anti_hbc"],
            },
            {
                "name": "anti_hbs",
                "header": "Anti-HBs",
                "path": ["serologies", "anti_hbs"],
            },
            {"name": "dgv_vhe", "header": "DGV VHE", "path": ["serologies", "dgv_vhe"]},
            {
                "name": "serologie_anguillulose",
                "header": "Sérologie Anguillulose",
                "path": ["serologies", "serologie_anguillulose"],
            },
            {
                "name": "syphilis_tpha",
                "header": "Syphilis TPHA",
                "path": ["serologies", "syphilis_tpha"],
            },
            {
                "name": "anti_cmv",
                "header": "Anti-CMV",
                "path": ["serologies", "anti_cmv"],
            },
            {
                "name": "anti_ebv",
                "header": "Anti-EBV",
                "path": ["serologies", "anti_ebv"],
            },
            {
                "name": "anti_toxoplasmose",
                "header": "Anti-Toxoplasmose",
                "path": ["serologies", "anti_toxoplasmose"],
            },
            {
                "name": "anti_hhv8",
                "header": "Anti-HHV8",
                "path": ["serologies", "anti_hhv8"],
            },
            # Morphology
            {
                "name": "perimetre_bi_mamelonnaire",
                "header": "Périmètre bi-mamelonnaire (cm)",
                "path": ["morphologie", "perimetre_bi_mamelonnaire"],
            },
            {
                "name": "hauteur_sternale",
                "header": "Hauteur sternale (cm)",
                "path": ["morphologie", "hauteur_sternale"],
            },
            {
                "name": "perimetre_ombilical",
                "header": "Périmètre ombilical (cm)",
                "path": ["morphologie", "perimetre_ombilical"],
            },
            # Habitus
            {
                "name": "alcoolisme",
                "header": "Alcoolisme",
                "path": ["habitus", "alcoolisme"],
            },
            {
                "name": "tabagisme",
                "header": "Tabagisme",
                "path": ["habitus", "tabagisme"],
            },
            {"name": "tabagisme_quantite", "header": "Tabagisme (PA)", "path": ["habitus", "tabagisme_quantite"]},
            
            {
                "name": "toxicomanie",
                "header": "Toxicomanie",
                "path": ["habitus", "toxicomanie"],
            },
            # Medical history
            {
                "name": "traitement",
                "header": "Traitement",
                "path": ["antecedents", "traitement"],
            },
            {
                "name": "hta",
                "header": "Hypertension artérielle",
                "path": ["antecedents", "hta"],
            },
            {
                "name": "diabete",
                "header": "Diabète",
                "path": ["antecedents", "diabete"],
            },
            {
                "name": "maladie_broncho_pulmonaire",
                "header": "Maladie broncho-pulmonaire",
                "path": ["antecedents", "maladie_broncho_pulmonanire"],
            },
            # Infectious assessment
            {
                "name": "antibiotherapie",
                "header": "Antibiothérapie",
                "path": ["bilan_infectieux", "antibiotherapie"],
            },
            {
                "name": "si_oui_preciser",
                "header": "Antibiotiques précisés",
                "path": ["bilan_infectieux", "si_oui_preciser"],
            },
            {
                "name": "diag_covid19",
                "header": "Diagnostic COVID-19",
                "path": ["bilan_infectieux", "diag_covid19"],
            },
            # Hemodynamic assessment
            {
                "name": "arret_cardiaque_recup",
                "header": "Arrêt cardiaque récupéré",
                "path": ["bilan_hemodynamique", "arret_cardiaque_recup"],
            },
            {
                "name": "concentre_globulaire",
                "header": "Concentré globulaire",
                "path": ["bilan_hemodynamique", "concentre_globulaire"],
            },
            {
                "name": "plasma_frais_congele",
                "header": "Plasma frais congelé",
                "path": ["bilan_hemodynamique", "plasma_frais_congele"],
            },
            {
                "name": "concentre_plaquettaire",
                "header": "Concentré plaquettaire",
                "path": ["bilan_hemodynamique", "concentre_plaquettaire"],
            },
            {
                "name": "albumine",
                "header": "Albumine",
                "path": ["bilan_hemodynamique", "albumine"],
            },
            {
                "name": "autres_medicaments",
                "header": "Autres médicaments",
                "path": ["bilan_hemodynamique", "autres_medicaments"],
            },
            # Hemodynamic evolution
            {
                "name": "dopamine",
                "header": "Dopamine (gamma.k/mn)",
                "path": ["evolution_hemodynamique", "dopamine"],
            },
            {
                "name": "dobutamine",
                "header": "Dobutamine (gamma.k/mn)",
                "path": ["evolution_hemodynamique", "dobutamine"],
            },
            {
                "name": "adrenaline",
                "header": "Adrénaline (mg/h)",
                "path": ["evolution_hemodynamique", "adrenaline"],
            },
            {
                "name": "noradrenaline",
                "header": "Noradrénaline (mg/h)",
                "path": ["evolution_hemodynamique", "noradrenaline"],
            },
            # Pulmonary assessment
            {
                "name": "traumatise_broncho_pulmonaire_actuel",
                "header": "Traumatisme broncho-pulmonaire actuel",
                "path": ["bilan_pulmonaire", "traumatise_broncho_pulmonaire_actuel"],
            },
            {
                "name": "lesion_pleurale_traumatique_actuelle",
                "header": "Lésion pleurale traumatique actuelle",
                "path": ["bilan_pulmonaire", "lesion_pleurale_traumatique_actuelle"],
            },
            {
                "name": "radiographie_thoraco_pulmonaire",
                "header": "Radiographie thoraco-pulmonaire",
                "path": ["bilan_pulmonaire", "radiographie_thoraco_pulmonaire"],
            },
            {
                "name": "aspirations_tracheo_bronchiques",
                "header": "Aspirations trachéo-bronchiques",
                "path": ["bilan_pulmonaire", "aspirations_tracheo_bronchiques"],
            },
            {
                "name": "prelevement_bacteriologique",
                "header": "Prélèvement bactériologique",
                "path": ["bilan_pulmonaire", "prelevement_bacteriologique"],
            },
            {
                "name": "fibroscopie_bronchique",
                "header": "Fibroscopie bronchique",
                "path": ["bilan_pulmonaire", "fibroscopie_bronchique"],
            },
            # Respiratory parameters

            {"name": "gds_ph", "header": "pH", "path": ["gds", "pH"]},
            {"name": "gds_paco2", "header": "PaCO2 (mmHg)", "path": ["gds", "PaCO2"]},
            {"name": "gds_pao2", "header": "PaO2 (mmHg)", "path": ["gds", "PaO2"]},
            {"name": "gds_co3h", "header": "CO3H- (mmol/l)", "path": ["gds", "CO3H"]},
            {"name": "gds_sao2", "header": "SaO2 (%)", "path": ["gds", "SaO2"]},
            {"name": "gds_peep", "header": "PEEP (cm H2O)", "path": ["gds", "PEEP"]},
            {"name": "gds_fio2", "header": "FiO2 (%)", "path": ["gds", "fio2_percentage"]},

            # Cardiac morphological assessment
            {
                "name": "fraction_d_ejection",
                "header": "Fraction d'éjection",
                "path": ["bilan_cardiaque_morphologique", "fraction_d_ejection"],
            },
            # Thorax assessment
            {
                "name": "epanchement_gazeux_droit",
                "header": "Épanchement gazeux droit",
                "path": ["thorax", "epanchement_gazeux_droit"],
            },
            {
                "name": "epanchement_gazeux_gauche",
                "header": "Épanchement gazeux gauche",
                "path": ["thorax", "epanchement_gazeux_gauche"],
            },
            {
                "name": "epanchement_liquidien_droit",
                "header": "Épanchement liquidien droit",
                "path": ["thorax", "epanchement_liquidien_droit"],
            },
            {
                "name": "epanchement_liquidien_gauche",
                "header": "Épanchement liquidien gauche",
                "path": ["thorax", "epanchement_liquidien_gauche"],
            },
            {
                "name": "atelectasie_droit",
                "header": "Atélectasie droite",
                "path": ["thorax", "atelectasie_droit"],
            },
            {
                "name": "atelectasie_gauche",
                "header": "Atélectasie gauche",
                "path": ["thorax", "atelectasie_gauche"],
            },
            {
                "name": "contusion_pulmonaire_droit",
                "header": "Contusion pulmonaire droite",
                "path": ["thorax", "contusion_pulmonaire_droit"],
            },
            {
                "name": "contusion_pulmonaire_gauche",
                "header": "Contusion pulmonaire gauche",
                "path": ["thorax", "contusion_pulmonaire_gauche"],
            },
            {
                "name": "infiltrat_droit",
                "header": "Infiltrat droit",
                "path": ["thorax", "infiltrat_droit"],
            },
            {
                "name": "infiltrat_gauche",
                "header": "Infiltrat gauche",
                "path": ["thorax", "infiltrat_gauche"],
            },
            {
                "name": "images_compatibles_avec_inhalation_droit",
                "header": "Images compatibles avec inhalation droite",
                "path": ["thorax", "images_compatibles_avec_inhalation_droit"],
            },
            {
                "name": "images_compatibles_avec_inhalation_gauche",
                "header": "Images compatibles avec inhalation gauche",
                "path": ["thorax", "images_compatibles_avec_inhalation_gauche"],
            },
            # File information
            {
                "name": "chemin_relatif",
                "header": "Chemin du fichier",
                "path": ["chemin_relatif"],
            },
            {
                "name": "date_extraction",
                "header": "Date d'extraction",
                "path": ["date_extraction"],
            },
        ]

    def export_csv(
        self,
        data_list: List[Dict[str, Any]],
        output_path: Union[str, Path],
        add_timestamp: bool = True,
    ) -> str:
        """
        Generate a CSV file from a list of donor data dictionaries.

        Args:
            data_list: List of donor data dictionaries
            output_path: Path where the CSV file will be saved
            add_timestamp: Whether to add a timestamp to the filename

        Returns:
            Path to the generated CSV file
        """
        # Prepare output path with timestamp if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_obj = Path(output_path)
            filename = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
            output_path = path_obj.parent / filename
        else:
            output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract column headers
        headers = [col["header"] for col in self.columns]

        # Prepare the CSV data
        csv_data = []
        for donor in data_list:
            row = {}
            for column in self.columns:
                value = self._get_nested_value(donor, column["path"])
                row[column["header"]] = self._format_value(value)
            csv_data.append(row)

        # Write the CSV file
        try:
            with open(output_path, "w", encoding=self.encoding, newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=headers, delimiter=self.delimiter
                )
                writer.writeheader()
                writer.writerows(csv_data)

            self.log(f"CSV file generated successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            self.log(f"Error generating CSV file: {e}", level=logging.ERROR)
            raise

    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """
        Get a value from a nested dictionary using a path list.

        Args:
            data: Dictionary to extract value from
            path: List of keys representing the path to the value

        Returns:
            Extracted value or empty string if not found
        """
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return ""

        return current

    def _format_value(self, value: Any) -> str:
        """
        Format a value for CSV output with improved decimal handling.
        """
        # Get the latest value if it's a list
        value = self.get_latest_value(value)

        # Handle different value types
        if value is None:
            return ""
        elif isinstance(value, float):
            # Pour les valeurs comme pH, garder 2 décimales
            if 6.0 <= value <= 8.0:  # Probable pH value
                return f"{value:.2f}".replace(".", ",")
            # Pour les autres valeurs, garder 1 décimale si significative
            elif value != int(value):
                return f"{value:.1f}".replace(".", ",")
            else:
                return str(int(value))
        elif isinstance(value, int):
            return str(value)
        else:
            # Convert to string and clean up
            return str(value).strip()


def generate_csv_filename(
    base_name: str = "donneurs_data",
    include_version: bool = True,
    version: str = "0.1.0",
) -> str:
    """
    Generate a standardized CSV filename with timestamp and optional version.

    Args:
        base_name: Base name for the file
        include_version: Whether to include version in filename
        version: Version string to include

    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if include_version:
        return f"{base_name}_{timestamp}_v{version}.csv"
    else:
        return f"{base_name}_{timestamp}.csv"

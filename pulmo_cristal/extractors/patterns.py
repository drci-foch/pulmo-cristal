"""
Patterns Module for pulmo-cristal package.

This module contains all regex patterns used for extracting information
from donor PDF documents. Patterns are organized by information category
and pre-compiled for better performance.
"""

import re

# Compile flags used in most patterns
DEFAULT_FLAGS = re.IGNORECASE | re.DOTALL

# Basic donor information patterns
DONOR_PATTERNS = {
    "num_cristal": re.compile(r"N°\s*CRISTAL\s*(\d+)", DEFAULT_FLAGS),
    "type_donneur": re.compile(
        r"Le donneur est\s*(.*?)(?:Site de décès|$)", DEFAULT_FLAGS
    ),
    "age": re.compile(r"(\d+)\s*ans", DEFAULT_FLAGS),
    "sexe": re.compile(r"Sexe\s*:?\s*([MF])", DEFAULT_FLAGS),
    "groupe_sanguin": re.compile(
        r"Groupe ABO\s.*?\s*((?:[ABO]|AB)\s*[\+\-])", DEFAULT_FLAGS
    ),
    "date_naissance": re.compile(
        r"Date de naissance\s.*?\s*(\d{2}/\d{2}/\d{4})", DEFAULT_FLAGS
    ),
    "taille": re.compile(r"Taille\s.*?\s*(\d+)\s*cm", DEFAULT_FLAGS),
    "poids": re.compile(r"Poids.*?(\d+(?:\.\d+)?)\s*kg", DEFAULT_FLAGS),
    "date_creation": re.compile(
        r"Date de création du dossier\s*:?\s*(\d{2}/\d{2}/\d{4})", DEFAULT_FLAGS
    ),
    "date_entree_bloc": re.compile(
        r"Date d'entrée au bloc\s*:?\s*(\d{2}/\d{2}/\d{4})", DEFAULT_FLAGS
    ),
    "date_clampage": re.compile(
        r"(?:Date de clampage|Date et heure du constat de décès)\s*:?\s*(\d{2}/\d{2}/\d{4})",
        DEFAULT_FLAGS,
    ),
    "heure_clampage": re.compile(
        r"(?:Date de clampage|Date et heure du constat de décès)\s*:?\s*\d{2}/\d{2}/\d{4}\s*:?\s*(\d{2}:\d{2})",
        DEFAULT_FLAGS,
    ),
    "etiologie": re.compile(r"Etiologie \(cause de décès\)([^\n]*)\n", DEFAULT_FLAGS),
    "duree_ventilation": re.compile(
        r"(?:Nombre de jours de ventilation|Durée de ventilation)\s+(\d+)",
        DEFAULT_FLAGS,
    ),
    "commentaire": re.compile(
        r"Durée de ventilation.*[\r\n]+Commentaire(.*)", DEFAULT_FLAGS
    ),
}

# Alternative donor patterns for cases where primary patterns fail
DONOR_ALT_PATTERNS = {
    "type_donneur_alt": re.compile(r"Le donneur est\s*:\s*([^\n]*?)\n", DEFAULT_FLAGS),
}

# HLA patterns
HLA_PATTERNS = {
    "hla_basic": re.compile(
        r"A1\s+A2\s+B1\s+B2\s+C1\s+C2\s+DR1\s+DR2[^\n]*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        DEFAULT_FLAGS,
    ),
    "hla_dqa": re.compile(r"DQA\s+DQA\s*\n\s*(\d+)\s+(\d+)", DEFAULT_FLAGS),
    "hla_dqb": re.compile(r"DQB\s+DQB\s*\n\s*(\d+)\s+(\d+)", DEFAULT_FLAGS),
    "hla_dp": re.compile(r"DP\s+DP\s*\n\s*(\d+)\s+(\d+)", DEFAULT_FLAGS),
}

# Serology patterns
SEROLOGY_PATTERNS = {
    "antigene_p24": re.compile(r"Antigène P 24\s*([^\n]+)", DEFAULT_FLAGS),
    "combine_hiv": re.compile(r"Combiné HIV\s*([^\n]+)", DEFAULT_FLAGS),
    "dgv_vih": re.compile(r"DGV VIH\s*([^\n]+)", DEFAULT_FLAGS),
    "dgv_vhc": re.compile(r"DGV VHC\s*([^\n]+)", DEFAULT_FLAGS),
    "dgv_vhb": re.compile(r"DGV VHB\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_htlv": re.compile(r"Anticorps anti-HTLV\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_hcv": re.compile(r"Anticorps anti-HCV\s*([^\n]+)", DEFAULT_FLAGS),
    "antigene_hbs": re.compile(r"Antigène HBs\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_hbc": re.compile(r"Anticorps anti-HBc\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_hbs": re.compile(r"Anticorps anti-HBs\s*([^\n]+)", DEFAULT_FLAGS),
    "dgv_vhe": re.compile(r"DGV VHE\s*([^\n]+)", DEFAULT_FLAGS),
    "serologie_anguillulose": re.compile(
        r"Sérologie Anguillulose\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "syphilis_tpha": re.compile(r"Syphilis: TPHA\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_cmv": re.compile(r"Anticorps anti-CMV\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_ebv": re.compile(r"Anticorps anti-EBV\s*([^\n]+)", DEFAULT_FLAGS),
    "anti_toxoplasmose": re.compile(
        r"Anticorps anti-Toxoplasmose\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "anti_hhv8": re.compile(r"Anticorps anti-HHV8\s*([^\n]+)", DEFAULT_FLAGS),
}

# Morphology patterns
MORPHOLOGY_PATTERNS = {
    "perimetre_bi_mamelonnaire": re.compile(
        r"Périmètre bi-mamelonnaire\s+(\d*)?(cm)?", DEFAULT_FLAGS
    ),
    "hauteur_sternale": re.compile(r"Hauteur sternale\s+(\d*)?(cm)?", DEFAULT_FLAGS),
    "perimetre_ombilical": re.compile(
        r"Périmètre ombilical\s+(\d*)?(cm)?", DEFAULT_FLAGS
    ),
}

# Lifestyle/habits patterns
HABITUS_PATTERNS = {
    "alcoolisme": re.compile(r"Alcoolisme\s*([^\n]+)", DEFAULT_FLAGS),
    "tabagisme": re.compile(r"Tabagisme\s*([^\n]+)", DEFAULT_FLAGS),
    "tabagisme_quantite": re.compile(r"Si oui, nombre de paquets-années\s*(\d+)", DEFAULT_FLAGS | re.IGNORECASE),
    "toxicomanie": re.compile(r"Toxicomanie\s*([^\n]+)", DEFAULT_FLAGS),
}

# Medical history patterns
ANTECEDENTS_PATTERNS = {
    "traitement": re.compile(
        r"Le patient suivait-il un traitement \?\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "hta": re.compile(r"Hyper tension artérielle\?\s*([^\n]+)", DEFAULT_FLAGS),
    "diabete": re.compile(r"Diabète\s*([^\n]+)", DEFAULT_FLAGS),
    "maladie_broncho_pulmonanire": re.compile(
        r"Maladie broncho-pulmonaire\s*([^\n]+)", DEFAULT_FLAGS
    ),
}

# Infectious assessment patterns
BILAN_INFECTIEUX_PATTERNS = {
    "antibiotherapie": re.compile(
        r"Antibiothérapie(?!\s+et résultats)\s+([^\n]+)", DEFAULT_FLAGS
    ),
    "si_oui_preciser": re.compile(r"Si oui, préciser\?([^\n]+)", DEFAULT_FLAGS),
    "diag_covid19": re.compile(r"Diagnostic COVID-19\s+([^\n]+)", DEFAULT_FLAGS),
}

# Hemodynamic assessment patterns
BILAN_HEMODYNAMIQUE_PATTERNS = {
    "arret_cardiaque_recup": re.compile(
        r"Arrêt cardiaque récupéré \?\s+([^\n]+)", DEFAULT_FLAGS
    ),
    "concentre_globulaire": re.compile(
        r"concentré globulaire\s+([\d+]+)", DEFAULT_FLAGS
    ),
    "plasma_frais_congele": re.compile(
        r"Plasma frais congelé\s+([\d+]+)", DEFAULT_FLAGS
    ),
    "concentre_plaquettaire": re.compile(
        r"Concentré plaquettaire\s+([\d+]+)", DEFAULT_FLAGS
    ),
    "albumine": re.compile(r"Albumine\s+([^\n]+)", DEFAULT_FLAGS),
    "autres_medicaments": re.compile(r"Autres médicaments ([^\n]+)", DEFAULT_FLAGS),
}

# Hemodynamic evolution patterns - these may have multiple values so we extract the last one
EVOLUTION_HEMODYNAMIQUE_PATTERNS = {
    # Dopamine - dernière valeur dans la ligne
    "dopamine": re.compile(r"dopamine[^\n]*?(\d+(?:\.\d+)?)\s*gamma\.k/mn(?:[^\n]*gamma\.k/mn\s*)*$", re.MULTILINE | DEFAULT_FLAGS),
    
    # Dobutamine - dernière valeur dans la ligne  
    "dobutamine": re.compile(r"dobutamine[^\n]*?(\d+(?:\.\d+)?)\s*gamma\.k/mn(?:[^\n]*gamma\.k/mn\s*)*$", re.MULTILINE | DEFAULT_FLAGS),
    
    # Adrénaline - dernière valeur dans la ligne
    "adrenaline": re.compile(r"adrénaline[^\n]*?(\d+(?:\.\d+)?)\s*mg/h(?:[^\n]*mg/h\s*)*$", re.MULTILINE | DEFAULT_FLAGS),
    
    # Noradrénaline - dernière valeur dans la ligne
    "noradrenaline": re.compile(r"noradrénaline[^\n]*?(\d+(?:\.\d+)?)\s*mg/h(?:[^\n]*mg/h\s*)*$", re.MULTILINE | DEFAULT_FLAGS),
}

# Pulmonary assessment patterns
BILAN_PULMONAIRE_PATTERNS = {
    "traumatise_broncho_pulmonaire_actuel": re.compile(
        r"Traumatisme broncho-pulmonaire actuel\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "lesion_pleurale_traumatique_actuelle": re.compile(
        r"Lésion pleurale traumatique actuelle\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "radiographie_thoraco_pulmonaire": re.compile(
        r"Radiographie thoraco-pulmonaire\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "aspirations_tracheo_bronchiques": re.compile(
        r"Aspirations trachéo-bronchiques\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "prelevement_bacteriologique": re.compile(
        r"Prélèvement bactériologique\s*([^\n]+)", DEFAULT_FLAGS
    ),
    "fibroscopie_bronchique": re.compile(
        r"Fibroscopie bronchique\s+(Oui|Non)", DEFAULT_FLAGS
    ),
}

# Respiratory parameters patterns - these may have multiple values so we extract the last one
# PARAMETRES_RESPIRATOIRES_PATTERNS = {
#     # Pattern pour pH avec support des décimales
#     "pH": re.compile(r"pH\s+[\d\s]+\s*(\d+\.\d+)(?:\s+(\d+\.\d+))?", DEFAULT_FLAGS),
    
#     # Pattern pour PaCO2 - dernière valeur
#     "PaCO2": re.compile(r"PaCO2\s+(?:[\d\s]*mmHg\s*)*(\d+(?:\.\d+)?)\s*mmHg", DEFAULT_FLAGS),
    
#     # Pattern pour PaO2 - dernière valeur  
#     "PaO2": re.compile(r"PaO2\s+(?:[\d\s]*mmHg\s*)*(\d+(?:\.\d+)?)\s*mmHg", DEFAULT_FLAGS),
    
#     # Pattern pour CO3H- - dernière valeur
#     "CO3H": re.compile(r"CO3H-\s+(?:[\d\s]*mmol/l\s*)*(\d+(?:\.\d+)?)\s*mmol/l", DEFAULT_FLAGS),
    
#     # Pattern pour SaO2 - dernière valeur
#     "SaO2": re.compile(r"SaO2\s+(?:[\d\s]*%\s*)*(\d+(?:\.\d+)?)\s*%", DEFAULT_FLAGS),
    
#     # Pattern pour PEEP - dernière valeur
#     "PEEP": re.compile(r"PEEP\s+(?:[\d\s]*cm\s*d'eau\s*)*(\d+(?:\.\d+)?)\s*cm\s*d'eau", DEFAULT_FLAGS),
    
#     "fio2_percentage": re.compile(
#         r"FiO2(?:<100|=100)?\s*:?\s*(?:pourcentage\s*:?)?\s*(\d+(?:\.\d+)?)\s*%", 
#         DEFAULT_FLAGS
#     ),

# }

# Cardiac morphological assessment patterns
BILAN_CARDIAQUE_PATTERNS = {
    "fraction_d_ejection": re.compile(r"Fraction d'éjection\s*([^\n]+)", DEFAULT_FLAGS),
}

# Thorax assessment patterns
THORAX_PATTERNS = {
    "epanchement_gazeux_droit": re.compile(
        r"Epanchement gazeux\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS
    ),
    "epanchement_gazeux_gauche": re.compile(
        r"Epanchement gazeux\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS
    ),
    "epanchement_liquidien_droit": re.compile(
        r"Epanchement liquidien\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS
    ),
    "epanchement_liquidien_gauche": re.compile(
        r"Epanchement liquidien\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS
    ),
    "atelectasie_droit": re.compile(r"Atélectasie\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS),
    "atelectasie_gauche": re.compile(r"Atélectasie\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS),
    "contusion_pulmonaire_droit": re.compile(
        r"Contusion pulmonaire\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS
    ),
    "contusion_pulmonaire_gauche": re.compile(
        r"Contusion pulmonaire\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS
    ),
    "infiltrat_droit": re.compile(r"Infiltrat\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS),
    "infiltrat_gauche": re.compile(r"Infiltrat\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS),
    "images_compatibles_avec_inhalation_droit": re.compile(
        r"Images compatibles avec une inhalation\s+(Oui|Non)\s+\w+", DEFAULT_FLAGS
    ),
    "images_compatibles_avec_inhalation_gauche": re.compile(
        r"Images compatibles avec une inhalation\s+\w+\s+(Oui|Non)", DEFAULT_FLAGS
    ),
}

# Groups all pattern dictionaries for easy access
ALL_PATTERNS = {
    "donor": DONOR_PATTERNS,
    "donor_alt": DONOR_ALT_PATTERNS,
    "hla": HLA_PATTERNS,
    "serologies": SEROLOGY_PATTERNS,
    "morphologie": MORPHOLOGY_PATTERNS,
    "habitus": HABITUS_PATTERNS,
    "antecedents": ANTECEDENTS_PATTERNS,
    "bilan_infectieux": BILAN_INFECTIEUX_PATTERNS,
    "bilan_hemodynamique": BILAN_HEMODYNAMIQUE_PATTERNS,
    "evolution_hemodynamique": EVOLUTION_HEMODYNAMIQUE_PATTERNS,
    "bilan_pulmonaire": BILAN_PULMONAIRE_PATTERNS,
    # "parametres_respiratoires": PARAMETRES_RESPIRATOIRES_PATTERNS,
    "bilan_cardiaque_morphologique": BILAN_CARDIAQUE_PATTERNS,
    "thorax": THORAX_PATTERNS,
}


def get_pattern_group(group_name: str) -> dict:
    """
    Get a specific pattern group by name.

    Args:
        group_name: Name of the pattern group

    Returns:
        Dictionary of compiled regex patterns

    Raises:
        KeyError: If the group name doesn't exist
    """
    if group_name not in ALL_PATTERNS:
        raise KeyError(
            f"Pattern group '{group_name}' not found. Available groups: {', '.join(ALL_PATTERNS.keys())}"
        )

    return ALL_PATTERNS[group_name]


def create_custom_pattern(pattern: str, flags: int = DEFAULT_FLAGS) -> re.Pattern:
    """
    Create a custom compiled regex pattern.

    Args:
        pattern: Regex pattern string
        flags: Regex compilation flags

    Returns:
        Compiled regex pattern
    """
    return re.compile(pattern, flags)

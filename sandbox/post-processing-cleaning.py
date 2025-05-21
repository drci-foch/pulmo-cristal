"""
Donor Data Processing Script

This script processes medical donor data from the Cristal database:
- Standardizes donor types
- Extracts and formats blood group information
- Standardizes medical condition fields
- Rounds numerical medical measurements
- Exports the processed data to Excel with current date in filename
"""

import pandas as pd
import numpy as np
import os
from typing import Optional
from datetime import datetime


def load_data(path: str, sep: str = ";") -> pd.DataFrame:
    """Load CSV data from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep=sep)


def standardize_donor_type(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize donor type values."""
    df["Type de donneur"] = df["Type de donneur"].apply(
        lambda x: "SME" if "SME" in x
        else "DDAC M3" if "DDAC M3" in x
        else "DDME" if "DDME" in x
        else x
    )
    return df


def extract_blood_group_components(df: pd.DataFrame) -> pd.DataFrame:
    """Extract blood group and Rhesus factor from combined field."""
    df[["Groupe", "Rhésus"]] = df["Groupe sanguin"].str.extract(r"([ABOabo]+)\s*([+-]*)")
    df["Groupe"] = df["Groupe"].str.upper()
    
    # Reorganize columns
    cols = list(df.columns)
    index = cols.index("Groupe sanguin")
    new_order = cols[:index+1] + ["Groupe", "Rhésus"] + cols[index+1:]
    # Remove duplicates while preserving order
    new_order = list(dict.fromkeys(new_order))
    df = df[new_order]
    
    # Remove the last 3 columns
    if df.shape[1] > 3:
        df = df.iloc[:, :-3]
    
    return df


def standardize_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Round weight values using custom rounding rule."""
    df["Poids (kg)"] = np.where(
        df["Poids (kg)"] % 1 >= 0.5, 
        np.ceil(df["Poids (kg)"]), 
        np.floor(df["Poids (kg)"])
    ).astype(int)
    return df


def standardize_binary_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize binary medical condition fields."""
    # Standardize cardiac arrest field
    df["Arrêt cardiaque récupéré"] = df["Arrêt cardiaque récupéré"].apply(
        lambda x: "Non" if "Non" in x else "Oui" if "Oui" in x else x
    )
    
    # Standardize bronchopulmonary trauma field
    df["Traumatisme broncho-pulmonaire actuel"] = df[
        "Traumatisme broncho-pulmonaire actuel"
    ].apply(
        lambda x: "Non" if "Non" in x
        else "Oui" if "Oui" in x
        else "NA" if "Non renseigné" in x
        else "NA"
    )
    
    # Standardize pleural trauma field
    df["Lésion pleurale traumatique actuelle"] = df[
        "Lésion pleurale traumatique actuelle"
    ].apply(
        lambda x: "Non" if "Non" in x
        else "Oui" if "Oui" in x
        else "NA" if "Non renseigné" in x
        else "NA"
    )
    
    return df


def round_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric medical measurements to 0 decimal places."""
    # Columns to round
    colonnes = [
        "PaCO2 (mmHg)",
        "PaO2 (mmHg)",
        "CO3H- (mmol/l)",
        "SaO2 (%)",
        "Fraction d'éjection",
    ]
    
    # Convert to numeric and round
    df[colonnes] = df[colonnes].round(0)
    df["Fraction d'éjection"] = pd.to_numeric(df["Fraction d'éjection"], errors='coerce')
    
    return df


def process_donor_data(input_path: str, output_path: str = None) -> Optional[pd.DataFrame]:
    """
    Main function to process donor data from input to output.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output Excel file. If None, a default name with current date will be used.
        
    Returns:
        Processed DataFrame or None if processing fails
    """
    try:
        # Generate default output path with current date
        if output_path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            output_path = f"EXTRACTION_AUTO_{current_date}.xlsx"
            
        print(f"Processing donor data from {input_path}")
        
        # Load data
        df = load_data(input_path)
        
        # Process data sequentially
        df = standardize_donor_type(df)
        df = extract_blood_group_components(df)
        df = standardize_weight(df)
        df = standardize_binary_fields(df)
        df = round_numeric_fields(df)
        
        # Save to Excel
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.to_excel(output_path, index=False)
        print(f"Data processed and saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None


if __name__ == "__main__":
    # Input file path
    input_file = r"C:\Users\benysar\Documents\GitHub\pulmo-cristal\donneurs_data_v0.1.0_20250521_103320.csv"
    
    # Process the data - output filename will include current date
    processed_df = process_donor_data(input_file)
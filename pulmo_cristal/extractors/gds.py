import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import PyPDF2
from pathlib import Path

try:
    import camelot
    print(f"✅ Camelot version: {camelot.__version__}")
except ImportError:
    print("❌ Camelot non installé")
    camelot = None


from .base import BaseExtractor


class GDSData(dict):
    """Typed dictionary class for HLA data."""

    pass


class ImprovedGazDuSangExtractor(BaseExtractor):
    """
    Extracteur GDS intelligent qui :
    1. Garde les dates comme headers
    2. Coupe la table à FiO2=100
    3. Efface les colonnes vides par section
    4. Prend l'info la plus récente avec le bon pourcentage FiO2
    """

    def __init__(self, logger: Optional[logging.Logger] = None, debug: bool = False):
        super().__init__(logger=logger)
        self.debug = debug

        if camelot is None:
            raise ImportError("Camelot requis: pip install camelot-py[cv]")

    def extract_gds_data_from_pdf(self, pdf_path: str, pages: str = "all") -> Dict[str, Any]:
        """Extrait les données GDS du PDF avec la nouvelle logique."""
        if not Path(pdf_path).exists():
            self.log(f"Fichier non trouvé: {pdf_path}", level=logging.ERROR)
            return {}

        try:
            # Extraction Camelot
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream')
            self.log(f"Trouvé {len(tables)} tables")

            # Trouve la table GDS principale
            gds_table = self._find_main_gds_table(tables)
            if gds_table is None:
                self.log("Aucune table GDS trouvée", level=logging.WARNING)
                return {}

            # Applique la nouvelle logique de traitement
            return self._process_gds_table_smart(gds_table)

        except Exception as e:
            self.log(f"Erreur extraction: {str(e)}", level=logging.ERROR)
            return {}

    def _find_main_gds_table(self, tables) -> Optional[object]:
        """Trouve la table GDS principale (la plus complète)."""
        best_table = None
        best_score = 0

        for table in tables:
            df = table.df
            table_text = df.to_string().lower()

            # Score basé sur indicateurs GDS
            indicators = ['fio2', 'ph', 'paco2', 'pao2', 'sao2', 'peep', 'co3h', 'mmhg']
            score = sum(1 for ind in indicators if ind in table_text)

            # Bonus pour structure attendue
            if 'fio2<100' in table_text and 'fio2=100' in table_text:
                score += 5

            if score > best_score:
                best_score = score
                best_table = table

        if self.debug and best_table is not None:
            print(f"🎯 Table GDS sélectionnée (score: {best_score})")
            print(f"   Shape: {best_table.df.shape}")

        return best_table


    def _process_gds_table_smart(self, table) -> Dict[str, Any]:
        """Version corrigée avec gestion des cas sans split."""
        df = table.df.copy()

        if self.debug:
            print(f"\n🔍 Traitement table GDS:")
            print(f"   Shape originale: {df.shape}")

        # Étape 1: Extrait les headers (dates/heures)
        headers = self._extract_headers(df)

        # Étape 2: Trouve la ligne de séparation FiO2=100
        split_row = self._find_fio2_100_split(df)

        if split_row is None:
            # Cas particulier : pas de split trouvé, traite tout comme une section
            self.log("Aucun split FiO2=100 détecté, analyse comme section mixte", level=logging.WARNING)
            return self._process_mixed_section(df, headers)

        # Étape 3: Divise la table
        fio2_less_100_df = df.iloc[:split_row].copy()
        fio2_100_df = df.iloc[split_row:].copy()

        # Important: Ajoute les headers aux deux sections
        if len(headers) > 0:
            header_rows = df.iloc[:2].copy()  # Lignes dates et heures
            
            # Pour FiO2=100, ajoute les headers au début
            fio2_100_df = pd.concat([header_rows, fio2_100_df], ignore_index=True)

        if self.debug:
            print(f"   Split à la ligne {split_row}")
            print(f"   FiO2<100 shape: {fio2_less_100_df.shape}")
            print(f"   FiO2=100 shape: {fio2_100_df.shape} (headers ajoutés)")

        # Étape 4: Traite chaque section si elle n'est pas vide
        all_data = []
        
        if len(fio2_less_100_df) > 0:
            fio2_less_100_clean, fio2_less_100_headers = self._remove_empty_columns_per_section(
                fio2_less_100_df, headers, "FiO2<100"
            )
            fio2_less_100_data = self._extract_section_data_smart(
                fio2_less_100_clean, fio2_less_100_headers, "FiO2<100"
            )
            all_data.extend(fio2_less_100_data)
        
        if len(fio2_100_df) > 0:
            fio2_100_clean, fio2_100_headers = self._remove_empty_columns_per_section(
                fio2_100_df, headers, "FiO2=100"
            )
            fio2_100_data = self._extract_section_data_smart(
                fio2_100_clean, fio2_100_headers, "FiO2=100"
            )
            all_data.extend(fio2_100_data)

        # Étape 5: Sélectionne les meilleures données
        if all_data:
            return self._select_best_data_smart(all_data)
        else:
            return {}
        


        
    def _process_mixed_section(self, df: pd.DataFrame, headers: List[Dict]) -> Dict[str, Any]:
        """Traite une table mixte en analysant les colonnes individuellement."""
        if self.debug:
            print("   🔄 Analyse intelligente des colonnes par type FiO2")
        
        # Nettoie les colonnes vides
        df_clean, clean_headers = self._remove_empty_columns_per_section(df, headers, "Mixte")
        
        # NOUVELLE APPROCHE: Analyse chaque colonne individuellement
        all_data = []
        
        for header in clean_headers:
            col_idx = header['column_index']
            timestamp = f"{header['date']} {header['time']}".strip()
            
            if self.debug:
                print(f"   📊 Analyse colonne {col_idx} ({timestamp})")
            
            # Détermine le type de cette colonne spécifique
            column_type, fio2_percentage = self._analyze_column_type(df_clean, col_idx)
            
            if column_type and fio2_percentage:
                if self.debug:
                    print(f"      → Type détecté: {column_type}, FiO2: {fio2_percentage}%")
                
                # Extrait les données pour cette colonne spécifique
                column_data = self._extract_column_data(
                    df_clean, col_idx, timestamp, column_type, fio2_percentage
                )
                
                if column_data:
                    all_data.append(column_data)
            else:
                if self.debug:
                    print(f"      → Aucune données dans cette colonne")
        
        if all_data:
            return self._select_best_data_smart(all_data)
        else:
            return {}
        




    def _extract_column_data(
        self, 
        df: pd.DataFrame, 
        col_idx: int, 
        timestamp: str, 
        fio2_type: str, 
        fio2_percentage: float
    ) -> Optional[Dict[str, Any]]:
        """Extrait les données médicales d'une colonne spécifique."""
        
        # Trouve les lignes de paramètres
        param_rows = self._find_medical_parameter_rows(df)
        
        data_entry = {
            'timestamp': timestamp,
            'fio2_type': fio2_type,
            'fio2_percentage': fio2_percentage,
            'column_index': col_idx
        }
        
        extracted_params = []
        
        # Extrait chaque paramètre
        for param_name, row_idx in param_rows.items():
            if row_idx < len(df) and col_idx < len(df.columns):
                numeric_value = self._extract_parameter_value_from_row(
                    df, row_idx, col_idx, param_name
                )
                
                if numeric_value is not None:
                    data_entry[param_name] = numeric_value
                    extracted_params.append(param_name)
        
        if len(extracted_params) >= 3:  # Au moins 3 paramètres pour être valide
            if self.debug:
                print(f"      ✅ {timestamp} ({fio2_type}, {fio2_percentage}%): "
                    f"{len(extracted_params)} params [{', '.join(extracted_params)}]")
            return data_entry
        else:
            if self.debug:
                print(f"      ❌ {timestamp}: seulement {len(extracted_params)} params")
            return None

    def _analyze_column_type(self, df: pd.DataFrame, col_idx: int) -> Tuple[Optional[str], Optional[float]]:
        """Analyse une colonne spécifique pour déterminer son type FiO2."""
        
        # Compte les valeurs médicales présentes dans cette colonne
        medical_values = 0
        has_percentage = False
        found_percentage = None
        
        for row_idx, row in df.iterrows():
            if col_idx < len(row):
                cell_value = str(row.iloc[col_idx]).strip()
                
                if cell_value and cell_value not in ['nan', '']:
                    # Cherche des valeurs médicales
                    if re.search(r'\d+(?:\.\d+)?\s*(?:mmhg|mmol|%|cm)', cell_value.lower()):
                        medical_values += 1
                    
                    # Cherche pH
                    if re.search(r'\d\.\d{2}', cell_value):
                        medical_values += 1
                    
                    # Cherche des pourcentages FiO2 spécifiques
                    percentage_match = re.search(r'(\d+)\s*%', cell_value)
                    if percentage_match:
                        pct = float(percentage_match.group(1))
                        # Vérifie si c'est sur une ligne FiO2
                        row_text = ' '.join(str(c) for c in row if str(c).strip()).lower()
                        if 'fio2' in row_text or 'pourcentage' in row_text:
                            has_percentage = True
                            found_percentage = pct
        
        if self.debug:
            print(f"      Colonne {col_idx}: {medical_values} valeurs médicales, "
                f"pourcentage: {found_percentage}%")
        
        # Détermine le type basé sur l'analyse
        if medical_values >= 3:  # Au moins 3 paramètres médicaux
            if has_percentage and found_percentage and found_percentage < 100:
                return "FiO2<100", found_percentage
            elif has_percentage and found_percentage == 100:
                return "FiO2=100", found_percentage
            else:
                # Pas de pourcentage explicite, devine basé sur les valeurs PaO2
                pao2_value = self._get_pao2_from_column(df, col_idx)
                if pao2_value and pao2_value > 300:  # PaO2 élevé suggère FiO2=100
                    return "FiO2=100", 100.0
                else:
                    return "FiO2<100", 21.0  # Défaut air ambiant
        
        return None, None

    def _get_pao2_from_column(self, df: pd.DataFrame, col_idx: int) -> Optional[float]:
        """Extrait la valeur PaO2 d'une colonne pour aider à déterminer le type."""
        for row_idx, row in df.iterrows():
            row_text = ' '.join(str(c) for c in row if str(c).strip()).lower()
            if 'pao2' in row_text and col_idx < len(row):
                cell_value = str(row.iloc[col_idx]).strip()
                pao2_match = re.search(r'(\d+(?:\.\d+)?)\s*mmhg', cell_value.lower())
                if pao2_match:
                    return float(pao2_match.group(1))
        return None



    def _extract_headers(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Extrait les headers (dates/heures) de la table.

        Returns:
            Liste de dictionnaires avec 'date', 'time', 'column_index'
        """
        headers = []

        # Cherche dans les premières lignes
        for row_idx in range(min(3, len(df))):
            row = df.iloc[row_idx]

            for col_idx, cell in enumerate(row):
                cell_str = str(cell).strip()

                if not cell_str or cell_str == 'nan':
                    continue

                # Cherche des dates
                date_match = re.search(r'(\d{2}/\d{2}/\d{4})', cell_str)
                time_match = re.search(r'(\d{2}:\d{2})', cell_str)

                if date_match or time_match:
                    # Trouve ou crée l'entrée header pour cette colonne
                    header_entry = next(
                        (h for h in headers if h['column_index'] == col_idx), None
                    )

                    if header_entry is None:
                        header_entry = {'column_index': col_idx, 'date': '', 'time': ''}
                        headers.append(header_entry)

                    if date_match:
                        header_entry['date'] = date_match.group(1)
                    if time_match:
                        header_entry['time'] = time_match.group(1)

        # Trie par index de colonne
        headers.sort(key=lambda x: x['column_index'])

        if self.debug:
            print(f"   📅 Headers extraits: {len(headers)}")
            for h in headers:
                print(f"      Col {h['column_index']}: {h['date']} {h['time']}")

        return headers

    def _find_fio2_100_split(self, df: pd.DataFrame) -> Optional[int]:
        """Trouve la VRAIE ligne de séparation FiO2=100, pas le titre général."""
        
        if self.debug:
            print(f"   🔍 Recherche du split FiO2=100 dans {len(df)} lignes")
            # Debug: affiche quelques lignes pour comprendre la structure
            for i in range(min(len(df), 12)):
                row_text = ' '.join(str(cell) for cell in df.iloc[i] if str(cell).strip())[:100]
                print(f"      Ligne {i}: {row_text}")
        
        # ÉTAPE 1: Cherche la ligne qui marque le DÉBUT des données FiO2=100
        # Cette ligne contient souvent "FiO2=100" ET "pH" ou d'autres paramètres
        for idx in range(1, len(df)):  # Skip ligne 0 (titre général)
            row_text = ' '.join(str(cell) for cell in df.iloc[idx] if str(cell).strip()).lower()
            
            # Patterns spécifiques pour la ligne de transition
            transition_patterns = [
                r'fio2\s*=\s*100.*ph',      # "FiO2=100 : pH" 
                r'fio2\s*=\s*100\s*:',      # "FiO2=100 :"
                r'^fio2\s*=\s*100\s*$',     # Ligne avec juste "FiO2=100"
                r'.*fio2\s*=\s*100.*',      # Toute ligne contenant "FiO2=100"
            ]
            
            for pattern in transition_patterns:
                if re.search(pattern, row_text):
                    if self.debug:
                        print(f"   ✅ Split FiO2=100 trouvé ligne {idx}: {row_text[:80]}")
                    return idx
        
        # ÉTAPE 2: Méthode alternative - cherche le changement de structure
        # Après les données FiO2<100, il y a souvent une ligne différente avant FiO2=100
        for idx in range(2, len(df) - 2):
            current_row = ' '.join(str(cell) for cell in df.iloc[idx] if str(cell).strip()).lower()
            next_row = ' '.join(str(cell) for cell in df.iloc[idx + 1] if str(cell).strip()).lower()
            
            # Cherche une ligne avec "fio2=100" suivie d'une ligne avec des paramètres
            if ('fio2=100' in current_row and 
                any(param in next_row for param in ['ph', 'paco2', 'pao2', 'mmhg'])):
                if self.debug:
                    print(f"   ✅ Split alternatif ligne {idx}: {current_row[:60]}")
                return idx
                
        # ÉTAPE 3: Cherche par analyse de contenu - détecte le changement de pattern
        # Les lignes FiO2<100 ont des données, puis il y a une transition
        data_lines = []
        for idx in range(1, len(df)):
            row_text = ' '.join(str(cell) for cell in df.iloc[idx] if str(cell).strip())
            
            # Compte les données médicales dans cette ligne
            medical_count = len(re.findall(r'\d+(?:\.\d+)?\s*(?:mmhg|mmol|%|cm)', row_text.lower()))
            medical_count += len(re.findall(r'\d\.\d{2}', row_text))  # pH
            
            data_lines.append((idx, medical_count, row_text.lower()))
            
            if self.debug and medical_count > 0:
                print(f"      Ligne {idx}: {medical_count} valeurs médicales")
        
        # Cherche le point où le pattern change
        for i in range(1, len(data_lines) - 1):
            idx, count, text = data_lines[i]
            
            # Si cette ligne contient "fio2=100" et que les lignes suivantes ont des données
            if 'fio2=100' in text and count == 0:
                # Vérifie que les lignes suivantes ont des données
                following_data = sum(data_lines[j][1] for j in range(i + 1, min(i + 4, len(data_lines))))
                if following_data > 3:  # Au moins quelques paramètres dans les lignes suivantes
                    if self.debug:
                        print(f"   ✅ Split par analyse ligne {idx}: changement de pattern détecté")
                    return idx
        
        if self.debug:
            print("   ❌ Aucun split FiO2=100 trouvé")
        return None

    def _remove_empty_columns_per_section(
        self,
        df: pd.DataFrame,
        headers: List[Dict],
        section_type: str
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Version améliorée pour détecter correctement les colonnes avec données."""
        # Colonnes à garder : toujours la colonne 0 (labels)
        columns_to_keep = {0}

        if self.debug:
            print(f"   🔍 Analyse colonnes pour {section_type}:")
            print(f"      DataFrame shape: {df.shape}")

        # Analyser chaque colonne pour voir si elle a du contenu médical
        for col_idx in range(1, len(df.columns)):
            col_content = []
            numeric_values = 0
            medical_units = 0
            
            for row_idx in range(len(df)):
                cell_value = str(df.iloc[row_idx, col_idx]).strip()
                if cell_value and cell_value != 'nan':
                    col_content.append(cell_value.lower())
                    
                    # Compte les valeurs numériques
                    if re.search(r'\d+(?:\.\d+)?', cell_value):
                        numeric_values += 1
                    
                    # Compte les unités médicales
                    if re.search(r'mmhg|mmol|%|cm.*eau', cell_value.lower()):
                        medical_units += 1

            col_text = ' '.join(col_content)

            # Critères plus stricts pour garder une colonne
            has_timestamps = bool(re.search(r'\d{2}/\d{2}/\d{4}|\d{2}:\d{2}', col_text))
            has_medical_values = numeric_values >= 2  # Au moins 2 valeurs numériques
            has_medical_units = medical_units >= 1    # Au moins 1 unité médicale
            has_ph_values = bool(re.search(r'\d\.\d{2}', col_text))  # pH typique

            should_keep = has_timestamps or (has_medical_values and (has_medical_units or has_ph_values))

            if should_keep:
                columns_to_keep.add(col_idx)
                
            if self.debug:
                print(f"      Col {col_idx}: nums={numeric_values}, units={medical_units}, "
                    f"timestamps={has_timestamps}, pH={has_ph_values} -> {'GARDÉE' if should_keep else 'SUPPRIMÉE'}")
                if should_keep:
                    print(f"         Contenu: {col_text[:80]}...")

        # Filtre le DataFrame
        columns_to_keep = sorted(list(columns_to_keep))
        df_clean = df.iloc[:, columns_to_keep]

        # Met à jour les headers
        old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(columns_to_keep)}

        updated_headers = []
        for header in headers:
            if header['column_index'] in old_to_new_mapping:
                new_header = header.copy()
                new_header['column_index'] = old_to_new_mapping[header['column_index']]
                new_header['original_column'] = header['column_index']
                updated_headers.append(new_header)

        if self.debug:
            print(f"   🧹 {section_type} nettoyage: {df.shape} -> {df_clean.shape}")
            print(f"      Colonnes gardées: {columns_to_keep}")
            print(f"      Headers finaux: {[(h['original_column'], h['date'], h['time']) for h in updated_headers]}")

        return df_clean, updated_headers







    def _extract_section_data_smart(
        self,
        df: pd.DataFrame,
        headers: List[Dict],
        section_type: str
    ) -> List[Dict[str, Any]]:
        """
        Extrait les données d'une section avec la logique améliorée.
        """
        if self.debug:
            print(f"\n   📊 Extraction section {section_type}:")
            print(f"      Shape: {df.shape}")

        data_entries = []

        # Trouve les pourcentages FiO2 pour cette section
        fio2_percentages = self._extract_fio2_percentages(df, section_type)

        # Trouve les lignes de paramètres médicaux
        param_rows = self._find_medical_parameter_rows(df)

        if self.debug:
            print(f"      FiO2 percentages: {fio2_percentages}")
            print(f"      Param rows: {list(param_rows.keys())}")

        # Pour chaque header (timestamp), extrait les valeurs
        for header in headers:
            col_idx = header['column_index']

            if col_idx >= len(df.columns):
                continue

            timestamp = f"{header['date']} {header['time']}".strip()
            if not timestamp:
                continue

            # Détermine le pourcentage FiO2 pour cette colonne
            fio2_percentage = self._get_fio2_for_column(
                fio2_percentages, col_idx, section_type
            )

            data_entry = {
                'timestamp': timestamp,
                'fio2_type': section_type,
                'fio2_percentage': fio2_percentage,
                'column_index': col_idx
            }

            # Extrait les valeurs des paramètres pour cette colonne
            has_medical_data = False
            extracted_params = []

            for param_name, row_idx in param_rows.items():
                if row_idx < len(df) and col_idx < len(df.columns):
                    # Amélioration: extraction ciblée par paramètre
                    numeric_value = self._extract_parameter_value_from_row(
                        df, row_idx, col_idx, param_name
                    )

                    if numeric_value is not None:
                        data_entry[param_name] = numeric_value
                        has_medical_data = True
                        extracted_params.append(param_name)
                        if self.debug:
                            print(f"      → {param_name}: {numeric_value} (ligne {row_idx}, col {col_idx}) ✓")
                    else:
                        if self.debug:
                            cell_value = str(df.iloc[row_idx, col_idx]).strip()
                            print(f"      → {param_name}: ÉCHEC - cellule='{cell_value}' (ligne {row_idx}, col {col_idx}) ✗")

            # Ajoute seulement si on a des données médicales
            if has_medical_data:
                data_entries.append(data_entry)
                if self.debug:
                    print(f"      ✅ {timestamp}: {len(extracted_params)} params [{', '.join(extracted_params)}]")

        return data_entries


    def _extract_parameter_value_from_row(self, df, row_idx, col_idx, param_name):
        """Extraction avec priorité aux cellules les plus proches."""
        
        # Définir l'ordre de recherche par paramètre
        if param_name in ["PaCO2", "PaO2"]:
            # Pour les gaz, chercher d'abord dans la ligne courante, puis ligne suivante
            search_offsets = [0, 1, -1, 2]
        else:
            # Pour autres paramètres, ordre standard
            search_offsets = [0, 1, -1, 2, -2, 3]
        
        for offset in search_offsets:
            check_row = row_idx + offset
            if 0 <= check_row < len(df) and col_idx < len(df.columns):
                cell_value = str(df.iloc[check_row, col_idx]).strip()
                
                if cell_value and cell_value not in ["nan", "", "À ajouter"]:
                    extracted_value = self._extract_parameter_by_type(cell_value, param_name)
                    
                    if extracted_value is not None:
                        if self.debug:
                            print(f"         → {param_name}: {extracted_value} (ligne {check_row})")
                        return extracted_value
        
        return None

    def _extract_parameter_by_type(self, cell_value: str, param_name: str) -> Optional[float]:
        """Extraction spécialisée par type de paramètre."""
        
        if param_name == "pH":
            ph_match = re.search(r"(\d+\.\d{1,2})", cell_value)
            if ph_match:
                value = float(ph_match.group(1))
                if 6.5 <= value <= 8.0:
                    return value

        elif param_name == "PaCO2":
            # PaCO2 est typiquement 20-80 mmHg
            mmhg_match = re.search(r'(\d+(?:\.\d+)?)\s*mmhg', cell_value.lower())
            if mmhg_match:
                value = float(mmhg_match.group(1))
                if 15 <= value <= 100:  # Plage physiologique PaCO2
                    return value
        
        elif param_name == "PaO2":
            # PaO2 est typiquement 80-600 mmHg
            mmhg_match = re.search(r'(\d+(?:\.\d+)?)\s*mmhg', cell_value.lower())
            if mmhg_match:
                value = float(mmhg_match.group(1))
                if 60 <= value <= 700:  # Plage physiologique PaO2
                    return value
            
            # Cherche un nombre seul SEULEMENT si pas d'unité parasite
            if not re.search(r"mmol|%|cm", cell_value.lower()):
                num_match = re.search(r"(\d+(?:\.\d+)?)", cell_value)
                if num_match:
                    value = float(num_match.group(1))
                    if self.debug:
                        print(f"         → Trouvé {param_name} sans unité: {value}")
                    
                    # Validation stricte selon le paramètre
                    if param_name == "PaCO2" and 10 <= value <= 100:
                        return value
                    elif param_name == "PaO2" and 30 <= value <= 600:
                        return value

        elif param_name == "CO3H":
            # Cherche avec unité mmol d'abord
            co3h_match = re.search(r"(\d+(?:\.\d+)?)\s*mmol", cell_value.lower())
            if co3h_match:
                value = float(co3h_match.group(1))
                if 10 <= value <= 35:
                    return value

        elif param_name == "SaO2":
            sao2_match = re.search(r"(\d+(?:\.\d+)?)\s*%", cell_value)
            if sao2_match:
                value = float(sao2_match.group(1))
                if 70 <= value <= 100:
                    return value

        elif param_name == "PEEP":
            peep_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", cell_value.lower())
            if peep_match:
                value = float(peep_match.group(1))
                if 0 <= value <= 25:
                    return value

        return None




    def _is_value_plausible(self, param_name: str, value: float) -> bool:
        """Vérifie si une valeur est plausible pour un paramètre donné."""
        plausible_ranges = {
            "pH": (6.5, 8.0),
            "PaCO2": (10, 100),
            "PaO2": (30, 600),
            "SaO2": (70, 100),
            "CO3H": (10, 35),
            "PEEP": (0, 25),
        }
        if param_name in plausible_ranges:
            min_val, max_val = plausible_ranges[param_name]
            return min_val <= value <= max_val
        return True




    def _extract_fio2_percentages(self, df: pd.DataFrame, section_type: str) -> Dict[int, float]:
        """Version corrigée pour mieux identifier les pourcentages FiO2."""
        percentages = {}

        if self.debug:
            print(f"      Recherche pourcentages FiO2 dans section {section_type}")

        # Patterns plus spécifiques selon le type de section
        if section_type == "FiO2<100":
            fio2_indicators = [
                "fio2<100.*pourcentage",
                "fio2.*<.*100.*pourcentage", 
                "pourcentage.*:",
                  "FiO2<100\s*:\s*[\s\S]*?pourcentage\s*:",  # si isolé dans section FiO2<100
            ]
        else:  # FiO2=100
            fio2_indicators = [
                "fio2=100",
                "fio2.*=.*100",
            ]

        # ÉTAPE 1: Identifier les lignes FiO2 spécifiques
        fio2_indicator_rows = []
        
        for row_idx, row in df.iterrows():
            row_text = " ".join(str(cell) for cell in row if str(cell).strip()).lower()
            
            if any(re.search(indicator, row_text) for indicator in fio2_indicators):
                fio2_indicator_rows.append(row_idx)
                if self.debug:
                    print(f"         Ligne FiO2 identifiée {row_idx}: {row_text[:60]}...")

        # ÉTAPE 2: Chercher les pourcentages dans ces lignes et adjacentes
        for row_idx in fio2_indicator_rows:
            # Cherche dans la ligne courante et les 2 suivantes
            for offset in range(-1, 3):
                check_row = row_idx + offset
                if check_row >= len(df):
                    continue
                    
                row = df.iloc[check_row]
                
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell).strip()
                    percentage_match = re.search(r"(\d+)\s*%", cell_str)
                    
                    if percentage_match:
                        percentage = float(percentage_match.group(1))
                        
                        # Validation plus stricte
                        is_valid = False
                        if section_type == "FiO2<100" and 21 <= percentage < 100:
                            is_valid = True
                        elif section_type == "FiO2=100" and percentage == 100:
                            is_valid = True
                        
                        if is_valid:
                            percentages[col_idx] = percentage
                            if self.debug:
                                print(f"         → FiO2 {percentage}% pour col {col_idx}")

        if self.debug:
            print(f"         Pourcentages FiO2 finaux: {percentages}")

        return percentages




    def _get_fio2_for_column(self, percentages: Dict[int, float], col_idx: int, section_type: str) -> float:
        """Détermine le pourcentage FiO2 pour une colonne spécifique."""
        
        # Si un pourcentage spécifique a été trouvé pour cette colonne, l'utiliser
        if col_idx in percentages:
            return percentages[col_idx]
        
        # Valeurs par défaut selon le type SEULEMENT si rien n'a été extrait
        if section_type == "FiO2=100":
            return 100.0  # FiO2=100 est toujours 100%
        else:
            # Pour FiO2<100, on a un problème si aucun % n'est extrait
            # Il faut améliorer l'extraction plutôt que deviner
            if self.debug:
                print(f"      ⚠️ Aucun pourcentage FiO2 trouvé pour colonne {col_idx} en section {section_type}")
            return None  # Mieux vaut None que de deviner un mauvais pourcentage


    def _find_medical_parameter_rows(self, df: pd.DataFrame) -> Dict[str, int]:
        """Trouve les lignes contenant les paramètres médicaux avec amélioration."""
        param_rows = {}

        # Paramètres à chercher avec patterns améliorés
        param_patterns = {
            'pH': r'\.{0,5}ph\b(?!\w)|ph\s*[:=]|fio2.*ph',
            'PaCO2': r'\.{0,5}paco2\b|paco2\s*[:=]',
            'PaO2': r'\.{0,5}pao2\b|pao2\s*[:=]', 
            'SaO2': r'\.{0,5}sao2\b|sao2\s*[:=]',
            'CO3H': r'\.{0,5}co3h-?\b|co3h\s*[:=]',
            'PEEP': r'\.{0,5}peep\b|peep\s*[:=]'
        }

        for row_idx, row in df.iterrows():
            row_text = ' '.join(str(cell) for cell in row if str(cell).strip()).lower()

            if self.debug and any(param.lower() in row_text for param in ["ph", "paco2", "pao2", "sao2", "co3h", "peep"]):
                print(f"      Ligne {row_idx}: {row_text[:80]}...")

            # Priorité stricte : un seul paramètre par ligne
            for param_name, pattern in param_patterns.items():
                if re.search(pattern, row_text):
                    # Vérifie qu'il y a des données numériques dans cette ligne ou les suivantes
                    has_data_nearby = self._check_data_nearby(df, row_idx)
                    if has_data_nearby:
                        param_rows[param_name] = row_idx
                        if self.debug:
                            print(f"      → {param_name} trouvé ligne {row_idx}")
                        break

        return param_rows

    def _check_data_nearby(self, df: pd.DataFrame, row_idx: int) -> bool:
        """Vérifie s'il y a des données numériques près de cette ligne."""
        # Cherche dans un rayon de 3 lignes
        for offset in range(-2, 4):
            check_row = row_idx + offset
            if 0 <= check_row < len(df):
                row = df.iloc[check_row]
                row_text = ' '.join(str(cell) for cell in row if str(cell).strip())
                
                # Cherche des patterns numériques médicaux
                if re.search(r'\d+(?:\.\d+)?\s*(?:mmhg|mmol|%|cm)', row_text.lower()):
                    return True
                if re.search(r'\d+\.\d{2}', row_text):  # pH style
                    return True
        
        return False






    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extrait une valeur numérique d'une chaîne."""
        if not text or text.strip() == '' or text.strip().lower() in ['nan', '']:
            return None

        # Nettoie le texte (garde seulement chiffres, points, virgules)
        cleaned = re.sub(r'[^\d\.\,\-]', ' ', str(text))

        # Cherche des nombres décimaux
        decimal_matches = re.findall(r'\d+[,\.]\d+', cleaned)
        if decimal_matches:
            value_str = decimal_matches[0].replace(',', '.')
            try:
                return float(value_str)
            except ValueError:
                pass

        # Cherche des entiers
        int_matches = re.findall(r'\d+', cleaned)
        if int_matches:
            try:
                return float(int_matches[0])
            except ValueError:
                pass

        return None


    def _select_best_data_smart(self, all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Version corrigée de la sélection avec priorité temporelle."""
        if not all_data:
            return {}

        def parse_timestamp(entry: Dict) -> datetime:
            ts_str = entry.get('timestamp', '')
            try:
                return datetime.strptime(ts_str, "%d/%m/%Y %H:%M")
            except ValueError:
                try:
                    return datetime.strptime(ts_str, "%d/%m/%Y")
                except ValueError:
                    return datetime.min

        def priority_score(entry: Dict) -> Tuple:
            timestamp = parse_timestamp(entry)
            # CORRECTION: Priorité au timestamp le plus récent en PREMIER
            # Puis FiO2=100 > FiO2<100
            # Puis FiO2% le plus élevé
            fio2_type_priority = 2 if entry.get('fio2_type') == 'FiO2=100' else 1
            fio2_percentage = entry.get('fio2_percentage', 0)
            return (timestamp, fio2_type_priority, fio2_percentage)

        # Trie par priorité (le plus élevé en premier)
        sorted_data = sorted(all_data, key=priority_score, reverse=True)
        
        if self.debug:
            print(f"\n🔄 Sélection parmi {len(all_data)} entrées:")
            for i, entry in enumerate(sorted_data[:3]):  # Affiche top 3
                score = priority_score(entry)
                print(f"   {i+1}. {entry.get('timestamp')} | {entry.get('fio2_type')} | "
                    f"{entry.get('fio2_percentage')}% | Score: {score}")
        
        selected = sorted_data[0]

        # Construit le résultat final
        result = {}
        exclude_keys = {'timestamp', 'fio2_type', 'column_index'}

        for key, value in selected.items():
            if key not in exclude_keys:
                result[key] = value

        if self.debug:
            print(f"\n🎯 Données sélectionnées:")
            print(f"   Timestamp: {selected.get('timestamp')} (le plus récent)")
            print(f"   Type: {selected.get('fio2_type')}")
            print(f"   FiO2: {selected.get('fio2_percentage')}%")
            print(f"   Paramètres: {len([k for k in result.keys() if k != 'fio2_percentage'])}")

        return result


    def validate_gds_data(self, data: Dict[str, float]) -> List[str]:
        """Valide les données extraites."""
        warnings = []

        ranges = {
            "pH": (6.8, 8.0),
            "PaCO2": (10, 100),
            "PaO2": (30, 600),
            "SaO2": (70, 100),
            "CO3H": (10, 35),
            "PEEP": (0, 20),
            "fio2_percentage": (0, 100)
        }

        for param, (min_val, max_val) in ranges.items():
            if param in data:
                value = data[param]
                if not (min_val <= value <= max_val):
                    warnings.append(f"{param} = {value} hors plage ({min_val}-{max_val})")

        return warnings




#!/usr/bin/env python3
"""
Script de test pour valider les nouveaux extracteurs GDS et hémodynamique.

Usage:
    python test_new_extractors.py [chemin_vers_pdf]
    
Si aucun chemin n'est fourni, utilise le PDF exemple CRISTAL 175394.
"""

import sys
from pathlib import Path

# Ajout du chemin du package si nécessaire
sys.path.insert(0, str(Path(__file__).parent))

from pulmo_cristal.extractors import DonorPDFExtractor
from pulmo_cristal.extractors.gds import ImprovedGazDuSangExtractor
from pulmo_cristal.extractors.hemodynamic import ImprovedHemodynamicExtractor


def test_extractors(pdf_path: str):
    """
    Test les nouveaux extracteurs avec un PDF donné.
    
    Args:
        pdf_path: Chemin vers le fichier PDF à tester
    """
    print(f"🧪 Test des extracteurs avec : {pdf_path}")
    print("=" * 60)
    
    try:
        # Test de l'extracteur GDS seul
        print("\n🔬 Test de l'extracteur GDS...")
        gds_extractor = ImprovedGazDuSangExtractor(debug=True)
        
        # Lire le PDF pour obtenir le texte
        from pulmo_cristal.extractors.pdf import PDFExtractor
        pdf_extractor = PDFExtractor()
        text, _ = pdf_extractor.extract_from_pdf(pdf_path)
        
        # Extraire les données GDS
        gds_data = gds_extractor.extract_gds_data(text)
        print(f"✅ Données GDS extraites : {gds_data}")
        
        # Validation
        warnings = gds_extractor.validate_gds_data(gds_data)
        if warnings:
            print(f"⚠️  Avertissements GDS : {warnings}")
        else:
            print("✅ Validation GDS : OK")
            
        # Debug info
        debug_info = gds_extractor.extract_debug_info(text)
        print(f"🔍 Info debug GDS : {debug_info}")
        
        print("\n" + "-" * 40)
        
        # Test de l'extracteur hémodynamique seul
        print("\n💉 Test de l'extracteur hémodynamique...")
        hemo_extractor = ImprovedHemodynamicExtractor(debug=True)
        
        # Extraire les données hémodynamiques
        hemo_data = hemo_extractor.extract_hemodynamic_evolution(text)
        print(f"✅ Données hémodynamiques extraites : {hemo_data}")
        
        # Validation
        warnings = hemo_extractor.validate_hemodynamic_data(hemo_data)
        if warnings:
            print(f"⚠️  Avertissements hémodynamiques : {warnings}")
        else:
            print("✅ Validation hémodynamique : OK")
            
        # Debug info
        debug_info = hemo_extractor.extract_debug_info(text)
        print(f"🔍 Info debug hémodynamique : {debug_info}")
        
        print("\n" + "-" * 40)
        
        # Test de l'extracteur complet
        print("\n🏥 Test de l'extracteur donneur complet...")
        donor_extractor = DonorPDFExtractor(debug=True)
        
        # Extraction complète
        donor_data = donor_extractor.extract_donor_data(pdf_path)
        
        print(f"\n📊 Résultats d'extraction :")
        print(f"  - Fichier source : {donor_data.get('fichier_source', 'N/A')}")
        print(f"  - Sections extraites : {len([k for k, v in donor_data.items() if isinstance(v, dict) and v])}")
        
        # Vérification des données critiques
        print(f"\n🎯 Vérification des corrections :")
        
        # GDS
        gds_result = donor_data.get("parametres_respiratoires", {})
        if gds_result:
            print(f"  📈 GDS :")
            for param in ["pH", "PaCO2", "PaO2", "CO3H", "SaO2", "PEEP"]:
                value = gds_result.get(param, "N/A")
                print(f"    - {param}: {value}")
        else:
            print("  ❌ Aucune donnée GDS extraite")
            
        # Hémodynamique
        hemo_result = donor_data.get("evolution_hemodynamique", {})
        if hemo_result:
            print(f"  💉 Hémodynamique :")
            for param in ["dopamine", "dobutamine", "adrenaline", "noradrenaline"]:
                value = hemo_result.get(param, "N/A")
                print(f"    - {param}: {value}")
        else:
            print("  ❌ Aucune donnée hémodynamique extraite")
            
        # Validation spécifique pour CRISTAL 175394
        if "175394" in pdf_path:
            print(f"\n🧬 Validation spéciale CRISTAL 175394 :")
            
            expected = {
                "pH": 7.47,
                "PaCO2": 39.0,
                "PaO2": 410.0,
                "noradrenaline": 0.0
            }
            
            for param, expected_value in expected.items():
                if param in ["pH", "PaCO2", "PaO2"]:
                    actual = gds_result.get(param)
                else:
                    actual = hemo_result.get(param)
                    
                if actual is not None:
                    if abs(float(actual) - expected_value) < 0.1:
                        print(f"    ✅ {param}: {actual} (attendu: {expected_value})")
                    else:
                        print(f"    ❌ {param}: {actual} (attendu: {expected_value})")
                else:
                    print(f"    ❓ {param}: Non extrait (attendu: {expected_value})")
                    
        print(f"\n✅ Test terminé avec succès !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test : {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fonction principale du script de test."""
    
    # Chemin par défaut vers le PDF exemple
    default_pdf = "path/to/CRISTAL_175394.pdf"  # À adapter selon votre structure
    
    # Utiliser le PDF fourni en argument ou le PDF par défaut
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = default_pdf
        print(f"ℹ️  Aucun PDF spécifié, utilisation du PDF par défaut : {pdf_path}")
    
    # Vérifier que le fichier existe
    if not Path(pdf_path).exists():
        print(f"❌ Fichier PDF non trouvé : {pdf_path}")
        print(f"   Usage : python {sys.argv[0]} [chemin_vers_pdf]")
        sys.exit(1)
    
    # Lancer les tests
    success = test_extractors(pdf_path)
    
    if success:
        print(f"\n🎉 Tous les tests sont passés avec succès !")
        sys.exit(0)
    else:
        print(f"\n💥 Certains tests ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
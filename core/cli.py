#!/usr/bin/env python3
"""
Interface en ligne de commande pour le système de détection de tableaux.

Permet d'utiliser le détecteur directement depuis le terminal avec des options avancées.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import logging

from src.detector import TableDetector
from src.exceptions import TableDetectionError


def setup_logging(verbose: bool = False):
    """Configure le logging selon le niveau de verbosité."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def detect_single_image(args) -> int:
    """Traiter une seule image."""
    print(f"🔍 Détection de tableaux dans: {args.image}")
    
    # Initialiser le détecteur
    detector = TableDetector(
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    try:
        # Charger le modèle
        print("📥 Chargement du modèle...")
        detector.load_model()
        print("✅ Modèle chargé")
        
        # Effectuer la détection
        result = detector.predict(args.image)
        
        if result.success:
            print(f"✅ Détection réussie!")
            print(f"📊 Trouvé {result.num_detections} tableau(x)")
            print(f"⏱️  Temps de traitement: {result.processing_time:.2f}s")
            
            # Afficher les détails
            for i, detection in enumerate(result.detections, 1):
                print(f"\n📋 Table {i}:")
                print(f"   Confiance: {detection.confidence:.3f}")
                print(f"   Position: {detection.bbox.to_list()}")
                print(f"   Aire: {detection.bbox.area:.1f} pixels²")
            
            # Sauvegarder les résultats si demandé
            if args.output:
                output_data = {
                    "image": str(args.image),
                    "success": result.success,
                    "detections": [
                        {
                            "bbox": detection.bbox.to_list(),
                            "confidence": detection.confidence,
                            "label": detection.label
                        }
                        for detection in result.detections
                    ],
                    "processing_time": result.processing_time,
                    "num_detections": result.num_detections
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"💾 Résultats sauvegardés dans: {args.output}")
            
            # Créer une visualisation si demandé
            if args.visualize:
                viz_path = args.visualize
                detector.visualize_predictions(
                    image_input=args.image,
                    output_path=viz_path,
                    show_confidence=True
                )
                print(f"🖼️  Visualisation sauvegardée dans: {viz_path}")
        else:
            print(f"❌ Détection échouée: {result.error_message}")
            return 1
            
    except TableDetectionError as e:
        print(f"❌ Erreur de détection: {e}")
        return 1
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return 1
    
    return 0


def detect_batch_images(args) -> int:
    """Traiter plusieurs images."""
    # Collecter les images
    images = []
    
    if args.directory:
        # Scanner un dossier
        image_dir = Path(args.directory)
        if not image_dir.exists():
            print(f"❌ Dossier introuvable: {args.directory}")
            return 1
        
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        for ext in extensions:
            images.extend(image_dir.glob(f"*{ext}"))
            images.extend(image_dir.glob(f"*{ext.upper()}"))
    elif args.images:
        # Utiliser la liste d'images fournie
        images = [Path(img) for img in args.images]
    else:
        print("❌ Vous devez spécifier soit --images soit --directory")
        return 1
    
    if not images:
        print("❌ Aucune image trouvée")
        return 1
    
    print(f"🔍 Traitement par lots de {len(images)} image(s)")
    
    # Initialiser le détecteur
    detector = TableDetector(
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    try:
        # Charger le modèle
        print("📥 Chargement du modèle...")
        detector.load_model()
        print("✅ Modèle chargé")
        
        # Traitement par lots
        batch_result = detector.predict_batch(images, max_batch_size=args.batch_size)
        
        # Afficher les résultats
        print(f"\n📊 Résultats du traitement par lots:")
        print(f"   Images traitées: {batch_result.total_images}")
        print(f"   Réussites: {batch_result.successful_detections}")
        print(f"   Échecs: {batch_result.failed_detections}")
        print(f"   Taux de réussite: {batch_result.success_rate:.1f}%")
        print(f"   Temps total: {batch_result.total_processing_time:.2f}s")
        print(f"   Temps moyen par image: {batch_result.avg_processing_time:.2f}s")
        
        # Détails par image si verbeux
        if args.verbose:
            print(f"\n📋 Détails par image:")
            for i, (result, img_path) in enumerate(zip(batch_result.results, images)):
                status = "✅" if result.success else "❌"
                detections = len(result.detections) if result.success else 0
                print(f"   {status} {img_path.name}: {detections} tableau(x)")
        
        # Sauvegarder les résultats
        if args.output:
            output_data = {
                "batch_summary": {
                    "total_images": batch_result.total_images,
                    "successful_detections": batch_result.successful_detections,
                    "failed_detections": batch_result.failed_detections,
                    "success_rate": batch_result.success_rate,
                    "total_processing_time": batch_result.total_processing_time
                },
                "results": []
            }
            
            for result, img_path in zip(batch_result.results, images):
                result_data = {
                    "image": str(img_path),
                    "success": result.success,
                    "processing_time": result.processing_time
                }
                
                if result.success:
                    result_data["detections"] = [
                        {
                            "bbox": detection.bbox.to_list(),
                            "confidence": detection.confidence,
                            "label": detection.label
                        }
                        for detection in result.detections
                    ]
                else:
                    result_data["error"] = result.error_message
                
                output_data["results"].append(result_data)
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Résultats sauvegardés dans: {args.output}")
        
    except TableDetectionError as e:
        print(f"❌ Erreur de détection: {e}")
        return 1
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return 1
    
    return 0


def benchmark_model(args) -> int:
    """Effectuer un benchmark du modèle."""
    print("⚡ Benchmark du modèle de détection")
    
    # Créer des images de test
    from PIL import Image, ImageDraw
    import tempfile
    import time
    import statistics
    
    temp_dir = Path(tempfile.mkdtemp())
    test_images = []
    
    # Générer des images de test de différentes tailles
    sizes = [(600, 400), (800, 600), (1200, 900), (1600, 1200)]
    
    print("🔧 Génération d'images de test...")
    for i, (width, height) in enumerate(sizes):
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Dessiner un tableau simple
        draw.rectangle([50, 50, width-50, height-50], outline='black', width=2)
        for j in range(3):
            y = 50 + (j * (height-100) // 3)
            draw.line([50, y, width-50, y], fill='black', width=1)
        
        img_path = temp_dir / f"test_{width}x{height}.png"
        img.save(img_path)
        test_images.append(img_path)
    
    # Initialiser le détecteur
    detector = TableDetector(device=args.device)
    
    try:
        detector.load_model()
        
        # Benchmark
        times = []
        print(f"\n🏃 Test de performance sur {len(test_images)} images...")
        
        for img_path in test_images:
            start_time = time.time()
            result = detector.predict(img_path)
            processing_time = time.time() - start_time
            times.append(processing_time)
            
            status = "✅" if result.success else "❌"
            detections = len(result.detections) if result.success else 0
            print(f"   {status} {img_path.name}: {processing_time:.2f}s, {detections} détection(s)")
        
        # Statistiques
        print(f"\n📊 Statistiques de performance:")
        print(f"   Temps moyen: {statistics.mean(times):.2f}s")
        print(f"   Temps médian: {statistics.median(times):.2f}s")
        print(f"   Temps min: {min(times):.2f}s")
        print(f"   Temps max: {max(times):.2f}s")
        if len(times) > 1:
            print(f"   Écart-type: {statistics.stdev(times):.2f}s")
        
        # Test de charge
        if args.stress_test:
            print(f"\n🔥 Test de charge (10 images simultanées)...")
            start_time = time.time()
            batch_result = detector.predict_batch(test_images * 3)  # 12 images
            total_time = time.time() - start_time
            
            print(f"   Temps total: {total_time:.2f}s")
            print(f"   Images/seconde: {batch_result.total_images / total_time:.2f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du benchmark: {e}")
        return 1
    finally:
        # Nettoyage
        import shutil
        shutil.rmtree(temp_dir)
    
    return 0


def main():
    """Fonction principale CLI."""
    parser = argparse.ArgumentParser(
        description="Détection de tableaux dans des documents avec DETR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s single invoice.jpg --visualize result.jpg
  %(prog)s batch --images *.png --output results.json
  %(prog)s --confidence 0.8 batch --directory ./images 
        """
    )
    
    # Arguments globaux
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Seuil de confiance (0.0-1.0, défaut: 0.7)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='Device à utiliser (défaut: cpu)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mode verbeux')
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande single
    single_parser = subparsers.add_parser('single', help='Traiter une seule image')
    single_parser.add_argument('image', type=Path, help='Chemin vers l\'image')
    single_parser.add_argument('--output', '-o', type=Path,
                              help='Fichier JSON pour sauvegarder les résultats')
    single_parser.add_argument('--visualize', type=Path,
                              help='Créer une visualisation avec boîtes englobantes')
    
    # Commande batch
    batch_parser = subparsers.add_parser('batch', help='Traiter plusieurs images')
    batch_group = batch_parser.add_mutually_exclusive_group(required=False)
    batch_group.add_argument('--images', nargs='+', help='Liste d\'images à traiter')
    batch_group.add_argument('--directory', type=Path,
                            help='Dossier contenant les images à traiter')
    batch_parser.add_argument('--output', '-o', type=Path,
                             help='Fichier JSON pour sauvegarder les résultats')
    batch_parser.add_argument('--batch-size', type=int, default=4,
                             help='Taille des lots pour le traitement (défaut: 4)')
    
    # Commande benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark du modèle')
    benchmark_parser.add_argument('--stress-test', action='store_true',
                                 help='Inclure un test de charge')
    
    # Parse des arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configuration du logging
    setup_logging(args.verbose)
    
    # Exécution des commandes
    try:
        if args.command == 'single':
            return detect_single_image(args)
        elif args.command == 'batch':
            return detect_batch_images(args)
        elif args.command == 'benchmark':
            return benchmark_model(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n⏹️  Opération interrompue par l'utilisateur")
        return 1


if __name__ == "__main__":
    sys.exit(main())
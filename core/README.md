# Table Detection System using DETR

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Test Technique Dataleon** - Système de détection de tableaux dans des documents utilisant un modèle DETR pré-entraîné.

Un système robuste pour détecter des tableaux dans des images de documents comme les factures et les relevés bancaires, utilisant le modèle transformer DETR (Detection Transformer).

## 🚀 Fonctionnalités

- **Détection de tableaux haute précision** utilisant le modèle `TahaDouaji/detr-doc-table-detection`
- **Support multi-formats** : JPEG, PNG, TIFF, BMP, WEBP
- **Traitement par lots** pour l'efficacité
- **Gestion d'erreurs robuste** avec validation d'entrée complète
- **Visualisation des résultats** avec boîtes englobantes
- **Seuil de confiance configurable**
- **Support GPU/CPU automatique**
- **API Pydantic** pour la validation des données
- **Suite de tests complète** avec pytest
- **Interface en ligne de commande (CLI)** simple et puissante

## 📋 Prérequis

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- PIL/Pillow 10.0+

## 🔧 Installation

### Installation standard

```bash
# Cloner le repository
git clone <repository-url>
cd Table_detector/core

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installation en mode développement (optionnel)
pip install -e .
```

### Installation avec GPU (CUDA)

```bash
# Installer PyTorch avec support CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Puis installer les autres dépendances
pip install -r requirements.txt
```

## 🎯 Utilisation rapide

### Utilisation en ligne de commande (CLI)

Le script CLI principal se trouve dans `core/cli.py`.

#### Détection sur une seule image

```bash
python core/cli.py single path/to/image.jpg --visualize result.jpg
```

#### Traitement par lots

```bash
python core/cli.py batch --directory ./images --output results.json
# ou
python core/cli.py batch --images img1.jpg img2.jpg --output results.json
```

#### Benchmark du modèle

```bash
python core/cli.py benchmark --stress-test
```

#### Options globales

- `--confidence 0.8` : seuil de confiance
- `--device cuda` : forcer l'utilisation du GPU
- `--verbose` : mode verbeux

#### Aide

```bash
python core/cli.py --help
```

### Utilisation en tant que bibliothèque Python

```python
from src.detector import TableDetector

# Initialiser le détecteur
detector = TableDetector(confidence_threshold=0.7)

# Charger le modèle
detector.load_model()

# Détecter les tableaux dans une image
result = detector.predict("path/to/invoice.jpg")

if result.success:
    print(f"Trouvé {result.num_detections} tableau(x)")
    for i, detection in enumerate(result.detections):
        print(f"Table {i+1}: confiance={detection.confidence:.3f}")
else:
    print(f"Erreur: {result.error_message}")
```

### Traitement par lots

```python
# Traiter plusieurs images
image_paths = ["invoice1.jpg", "invoice2.jpg", "bank_statement.png"]
batch_result = detector.predict_batch(image_paths)

print(f"Traité {batch_result.total_images} images")
print(f"Taux de réussite: {batch_result.success_rate:.1f}%")
```

### Visualisation

```python
# Créer une visualisation avec boîtes englobantes
detector.visualize_predictions(
    image_input="invoice.jpg",
    output_path="result_with_boxes.jpg",
    show_confidence=True
)
```

## 📊 Exemples d'utilisation

Le dossier `examples/` contient des scripts détaillés :

- `basic_usage.py` : Utilisation de base et gestion d'erreurs
- `batch_processing.py` : Traitement par lots avec analyse des résultats

```bash
# Exécuter les exemples
python examples/basic_usage.py
python examples/batch_processing.py
```

## 🧪 Tests

Le système inclut une suite de tests complète couvrant tous les scénarios :

### Exécuter tous les tests

```bash
pytest tests/ -v
```

### Tests avec couverture

```bash
pytest tests/ --cov=src --cov-report=html
```

### Tests par catégorie

```bash
# Tests unitaires uniquement
pytest tests/ -m "unit"

# Tests d'intégration uniquement  
pytest tests/ -m "integration"

# Exclure les tests lents
pytest tests/ -m "not slow"
```

### Scénarios de test couverts

✅ **Tests de succès :**
- Détection réussie sur factures
- Détection réussie sur documents bancaires
- Traitement par lots
- Différents formats d'image

❌ **Tests d'erreur :**
- Fichier introuvable
- Image corrompue
- Format non supporté
- Image trop grande
- Pas de tableau détecté

⚡ **Tests de performance :**
- Temps de traitement
- Utilisation mémoire
- Requêtes concurrentes

## 📝 Configuration

### Variables d'environnement

Créer un fichier `.env` pour personnaliser la configuration :

```bash
MODEL_NAME=TahaDouaji/detr-doc-table-detection
CONFIDENCE_THRESHOLD=0.7
DEVICE=auto
MAX_IMAGE_SIZE=10485760
LOG_LEVEL=INFO
```

### Configuration par code

```python
from src.config import settings

# Modifier la configuration
settings.confidence_threshold = 0.8
settings.device = "cuda"
```

## 🔍 API Reference

### Classe principale : `TableDetector`

```python
class TableDetector:
    def __init__(
        self,
        model_name: str = "TahaDouaji/detr-doc-table-detection",
        confidence_threshold: float = 0.7,
        device: str = "auto"
    )
    
    def load_model(self) -> None
    def predict(self, image_input: Union[str, Path, Image.Image]) -> DetectionResult
    def predict_batch(self, images: List[...]) -> BatchDetectionResult
    def visualize_predictions(self, image_input, output_path: str) -> None
    def update_confidence_threshold(self, threshold: float) -> None
```

### Modèles de données

```python
class DetectionResult:
    success: bool
    detections: List[Detection]
    processing_time: float
    image_info: Optional[ImageInfo]
    error_message: Optional[str]

class Detection:
    bbox: BoundingBox
    confidence: float
    label: str

class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
```

## 🚨 Gestion d'erreurs

Le système gère plusieurs types d'erreurs :

- `ModelLoadError` : Échec du chargement du modèle
- `ImageProcessingError` : Erreur de traitement d'image
- `InvalidImageError` : Image invalide ou corrompue
- `UnsupportedFormatError` : Format d'image non supporté
- `ImageTooLargeError` : Image trop grande
- `PredictionError` : Erreur de prédiction du modèle

## 🎛️ Paramètres du modèle

### Seuil de confiance

- **0.5** : Plus de détections, moins précises
- **0.7** : Équilibre recommandé
- **0.9** : Détections très précises, moins nombreuses

### Formats supportés

- **JPEG/JPG** : Recommandé pour photos/scans
- **PNG** : Recommandé pour documents
- **TIFF** : Documents haute qualité
- **BMP** : Images non compressées
- **WEBP** : Format moderne

## ⚡ Performance

### Temps de traitement typiques

- **CPU** : 2-5 secondes par image
- **GPU** : 0.5-1 seconde par image

### Optimisations

- Utilisation automatique du GPU si disponible
- Traitement par lots efficace
- Gestion mémoire optimisée
- Cache du modèle

## 🐛 Dépannage

### Problèmes possibles

**Erreur de chargement du modèle :**
```bash
# Vérifier la connexion internet
# Le modèle sera téléchargé automatiquement
```

**Erreur CUDA :**
```bash
# Forcer l'utilisation du CPU
detector = TableDetector(device="cpu")
```

**Erreur de mémoire :**
```bash
# Réduire la taille du batch
batch_result = detector.predict_batch(images, max_batch_size=2)
```

## 🔄 Développement

### Qualité du code

```bash
# Formatage
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Structure du projet

```
Table_detector/
├── core/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── detector.py      # Classe principale
│   │   ├── models.py        # Modèles Pydantic
│   │   ├── config.py        # Configuration
│   │   └── exceptions.py    # Exceptions
│   ├── tests/
│   │   ├── conftest.py      # Configuration pytest
│   │   ├── test_detector.py # Tests principaux
│   │   └── test_models.py   # Tests des modèles
│   ├── examples/
│   │   ├── basic_usage.py
│   │   └── batch_processing.py
│   ├── cli.py               # Interface CLI
│   └── requirements.txt
```

## 📈 Métriques

Le système fournit des métriques détaillées :

- Temps de traitement par image
- Taux de succès/échec
- Statistiques de confiance
- Utilisation mémoire
- Nombre de détections par image

## 🙏 Remerciements

- **Dataleon** pour l'opportunité de ce test technique

---

**Développé pour le test technique Dataleon - Démonstration de compétences en Python, ML et Computer Vision**
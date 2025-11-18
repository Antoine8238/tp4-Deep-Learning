# TP4 : Advanced Vision, Segmentation, and 3D Data

##  Description

Implémentation de l'architecture U-Net pour la segmentation sémantique d'images médicales et introduction aux convolutions 3D pour données volumétriques. Ce projet applique les meilleures pratiques MLOps avec tracking des expériences via MLflow.

**Module :** Deep Learning Engineering - 5GI  
**École :** ENSPY, Université de Yaoundé I  
**Année :** 2024-2025

---

##  Objectifs

-  Maîtriser la segmentation sémantique avec U-Net
-  Implémenter des métriques spécifiques (IoU, Dice Coefficient)
-  Appliquer les pratiques MLOps (experiment tracking avec MLflow)
-  Comprendre les convolutions 3D pour données volumétriques
- Gérer les défis des données médicales (déséquilibre, taille limitée)

---

##  Structure du Projet

```
tp4-segmentation/
├── unet.py       # Architecture U-Net complète
├── metrics.py    # Métriques de segmentation (Dice, IoU)
├── train.py        # Entraînement avec MLflow tracking
├── conv3d.py       # Convolutions 3D pour données volumétriques
├── requirements.txt           # Dépendances Python


---

##  Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```
tensorflow>=2.13.0
mlflow>=2.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

##  Utilisation

### Exercice 2.1 : Architecture U-Net

Construction de l'architecture U-Net avec skip connections.

```bash
python exercice_2_1_unet.py
```

**Sortie attendue :**
- Résumé de l'architecture
- Nombre de paramètres
- Dimensions input/output

### Exercice 2.2 : Métriques de Segmentation

Implémentation et test des métriques Dice Coefficient et IoU.

```bash
python exercice_2_2_metrics.py
```

**Sortie attendue :**
- Tests sur données synthétiques
- Comparaison Dice vs IoU
- Validation des formules

### Exercice 2.3 : Entraînement avec MLflow

Génération de données synthétiques et entraînement du modèle U-Net.

```bash
python exercice_2_train.py
```

**Sortie attendue :**
- Génération de 200 images synthétiques
- Entraînement pendant 30 epochs (avec early stopping)
- Métriques finales (Dice, IoU, Loss)
- Tracking automatique dans MLflow

### Exercice 3 : Convolutions 3D

Implémentation d'un bloc Conv3D pour données volumétriques.

```bash
python exercice_3_conv3d.py
```

**Sortie attendue :**
- Architecture Conv3D
- Simulation d'entraînement
- Tracking MLflow

---

##  Visualisation des Résultats (MLflow)

Après avoir exécuté les scripts, visualisez les expériences :

```bash
mlflow ui
```

Puis ouvrez dans votre navigateur : **http://localhost:5000**

### Ce que vous verrez dans MLflow :

- **Expériences :** Toutes les runs trackées
- **Paramètres :** Architecture, optimizer, loss function, etc.
- **Métriques :** Courbes de convergence (Dice, IoU, Loss)
- **Artefacts :** Configuration des modèles (JSON)

---

##  Architecture U-Net

```
Input (128×128×1)
      ↓
[Conv Block 32] ────────────────────┐
      ↓ MaxPool (64×64)              │
[Conv Block 64] ──────────────┐     │
      ↓ MaxPool (32×32)        │     │
[Conv Block 128] ───────┐     │     │
      ↓ MaxPool (16×16) │     │     │
[Conv Block 256]        │     │     │
      ↓ Upsample         │     │     │
[Concat + Conv 128] ←───┘     │     │  Skip Connections
      ↓ Upsample               │     │
[Concat + Conv 64]  ←─────────┘     │
      ↓ Upsample                     │
[Concat + Conv 32]  ←───────────────┘
      ↓
Output (128×128×1)
```

**Caractéristiques :**
- 4 niveaux (encoder + decoder)
- Skip connections par concatenation
- Activation sigmoid en sortie (segmentation binaire)
- ~X,XXX,XXX paramètres

---

##  Métriques Implémentées

### Dice Coefficient

```
Dice = 2·|A ∩ B| / (|A| + |B|)
```

- **Plage :** [0, 1] (1 = parfait)
- **Usage :** Métrique principale en imagerie médicale
- **Avantage :** Tolérant aux données déséquilibrées

### IoU (Intersection over Union)

```
IoU = |A ∩ B| / |A ∪ B|
```

- **Plage :** [0, 1] (1 = parfait)
- **Usage :** Segmentation générale
- **Avantage :** Plus strict que Dice

### Relation entre Dice et IoU

```
Dice = 2·IoU / (1 + IoU)
```

**Exemple :**
- IoU = 0.5 → Dice = 0.667
- IoU = 0.8 → Dice = 0.889

---

## Convolutions 3D

### Différence Conv2D vs Conv3D

| Aspect | Conv2D | Conv3D |
|--------|--------|--------|
| **Kernel** | k×k | k×k×k |
| **Paramètres (k=3)** | 9 | 27 |
| **Mouvement** | Sur (H, W) | Sur (D, H, W) |
| **Input** | H×W×C | D×H×W×C |

### Défis Computationnels

- **Mémoire :** ~27× plus coûteux qu'un Conv2D (kernel 3×3×3)
- **Solutions :**
  - Réduire la taille du kernel (3×3×3 → 2×2×2)
  - Moins de filtres par couche
  - Réduire la profondeur D (moins de slices)
  - Mixed precision training (float16)

---

## Résultats Attendus

### U-Net Training

| Métrique | Valeur |
|----------|--------|
| **Dice Coefficient** | > 0.85 |
| **IoU** | > 0.75 |
| **Loss** | < 0.20 |
| **Epochs** | ~15-25 (avec early stopping) |

*Remplacez par vos résultats réels après entraînement*

---

##  Configuration MLflow

### Convention de Nommage

Format : `{Architecture}_{Loss}_{Optimizer}`

**Exemples :**
- `UNet_DiceLoss_Adam`
- `UNet_CombinedLoss_SGD`
- `Conv3D_Baseline`

### Hyperparamètres Loggés

- Architecture
- Optimizer (type + learning rate)
- Loss function
- Batch size
- Nombre d'epochs

### Métriques Loggées

- Dice Coefficient (par epoch)
- IoU (par epoch)
- Loss (train + validation)
- Métriques finales

---

##  Questions Théoriques (Réponses dans le Rapport)

1. **Output de segmentation sémantique :** Dimension et nature du tenseur de sortie
2. **Skip connections U-Net vs ResNet :** Différence et rôle du decoder
3. **Loss functions pour données médicales :** Pourquoi cross-entropy est inadéquate
4. **Conv3D vs Conv2D :** Différences et nécessité pour données volumétriques
5. **Trade-offs Conv3D :** Gestion des contraintes mémoire

---

##  Technologies Utilisées

- **TensorFlow/Keras** : Framework Deep Learning
- **MLflow** : Experiment tracking et MLOps
- **NumPy** : Manipulation de données
- **Scikit-learn** : Train/test split
- **Python 3.8+** : Langage de programmation

---

##  Références

1. Ronneberger, O., et al. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** *MICCAI*.
2. Milletari, F., et al. (2016). **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.** *3DV*.
3. Sudre, C. H., et al. (2017). **Generalised Dice overlap as a deep learning loss function.** *DLMIA*.

---

## 👥 Auteur

**[Antoine Emmanuel ESSOMBA ESSOMBA]**  
Matricule : [23p750]  
Email : [essombantoine385@gmail.com]

---

## 📄 Licence

Ce projet est réalisé dans le cadre du module Deep Learning Engineering à l'ENSPY.

---




---

##  Support

Pour toute question ou problème :
1. Consultez le rapport PDF pour les détails théoriques
2. Vérifiez les logs MLflow pour les résultats d'expériences
3. Ouvrez une issue sur ce repository

---

**Dernière mise à jour :** Novembre 2025

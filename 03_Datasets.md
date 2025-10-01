# Les Datasets

### Table des matières

1. [Définiiton](#définition)
2. [Rôle des Datasets avec les LLM](#role-des-datasets-avec-les-llm)
3. [Types de datasets (pré-entraînement, fine-tuning)](#types-de-datasets-pré-entraînement-fine-tuning)
4. [Qualité des Datasets](#qualité-des-datasets)]
5. [Préparation et nettoyage des données](#préparation-et-nettoyage-des-données)
6. [Exemples d'accès aux datasets de HF](#exemples-dacces-aux-datasets-de-hf)
7. [Sources populaires (Hugging Face Hub, Common Crawl)](#sources-populaires-hugging-face-hub-common-crawl)

---
### Définition

    - Un dataset est une collection structurée de données textuelles utilisée pour entraîner, valider ou tester un modèle de langage (composés de textes variés : livres, articles, conversations, pages web, code source, etc.)    

---

### Rôle des Datasets avec les LLM

    - Entraînement (Training)
        - Les LLM apprennent des motifs linguistiques à partir des données textuelles.
        - Plus le dataset est large et diversifié, plus le modèle a de chances de généraliser et de comprendre différentes formes d’expression.

    - Validation
        - On utilise un jeu de données distinct pour vérifier que le modèle n’a pas juste « mémorisé » le dataset d’entraînement.
        - Cela permet de détecter l’overfitting (surapprentissage).

    - Test
        - Un troisième jeu de données, jamais vu par le modèle, permet d’évaluer ses performances « en conditions réelles ».

---

### Types de datasets (pré-entraînement, fine-tuning)

    - Données ouvertes : Wikipedia, Common Crawl, The Pile, OpenWebText, etc.
    - Données spécialisées : corpus médicaux, juridiques, techniques…
    - Données générées : données synthétiques créées automatiquement pour des tâches spécifiques.

---

### Qualité des Datasets

    - Pertinence : Un dataset de qualité améliore la compréhension du langage et la capacité du modèle à répondre correctement.
    - Biais : Les biais présents dans les données se retrouvent dans le modèle.
    - Sécurité : Datasets mal filtrés peuvent introduire des propos toxiques ou des erreurs.

---

### Préparation et nettoyage des données

- Préparer les données
    - Rassembler les données
        - Collecte depuis différentes sources : fichiers CSV, bases de données SQL, APIs, web scraping, etc.
        - Fusionner les différentes sources si besoin.
    - Vérifier la structure
        - Vérifier que les colonnes attendues sont présentes.
        - Uniformiser les formats (dates, nombres, textes…).
    - Échantillonnage
        - Prendre un sous-ensemble si les données sont trop volumineuses.
        - Séparer en jeux d’entraînement, validation, et test.

- Nettoyage des données  
    - a) Gestion des valeurs manquantes  
        - Supprimer les lignes ou colonnes trop incomplètes.
        - Imputer les valeurs: remplir avec la moyenne, la médiane, une valeur spécifique, ou par interpolation.
    - b) Détection et gestion des doublons    
         - Supprimer les lignes identiques ou similaires.
    - c) Correction des incohérences  
         - Uniformiser la casse ("Paris" et "paris" → "Paris").
         - Corriger les fautes de frappe, les abréviations non standardisées, etc.
    - d) Nettoyage du texte (spécifique NLP/LLM)  
         - Retirer: balises HTML, caractères spéciaux, espaces multiples, emojis ou symboles non désirés.
         - Normaliser: accents, ponctuation, casse.
         - Tokeniser le texte si besoin.
         - Filtrer les langues, retirer les textes trop courts ou non pertinents.
    - e) Filtrage de contenu  
         - Éliminer les textes offensants, sensibles ou non adaptés au contexte souhaité.
         - Supprimer le spam, les publicités, les parties non textuelles (code, logs, etc. selon le besoin).
    - f) Conversion des types  
         - Assurer que les colonnes numériques sont bien en float/int, les dates en datetime, etc.

    - Code  

---
### Exemples d'accès aux Datasets HuggingFace

#### 1. Installation (si nécessaire)
```bash
pip install datasets
```

#### 2. Charger un dataset public
```python
from datasets import load_dataset

# Exemple : Charger le dataset 'imdb' (sentiment analysis)
dataset = load_dataset("imdb")
print(dataset)
# Affiche : {'train': Dataset(...), 'test': Dataset(...), 'unsupervised': Dataset(...)}

# Accéder aux données d'entraînement
train_data = dataset["train"]
print(train_data[0])  # Affiche le premier exemple
```

#### 3. Charger un dataset multilingue
```python
from datasets import load_dataset

# Exemple : 'xnli' contient des données de classification en plusieurs langues
dataset = load_dataset("xnli")
print(dataset["train"][0])
```

#### 4. Filtrer et manipuler les données
```python
# Filtrer les exemples positifs dans IMDB
positive_reviews = dataset["train"].filter(lambda x: x["label"] == 1)
print(positive_reviews)
```

#### 5. Télécharger un dataset local ou depuis un fichier
```python
# Depuis un fichier CSV local
dataset = load_dataset("csv", data_files={"train": "chemin/vers/train.csv", "test": "chemin/vers/test.csv"})
print(dataset)
```

#### 6. Utiliser un dataset pour le batch processing
```python
# Prendre un batch de 5 exemples
batch = dataset["train"].select(range(5))
print(batch)
```

#### 7. Explorer les colonnes et la structure
```python
print(dataset["train"].features)
# Affiche les noms et types des colonnes/features disponibles
```

#### 8. Utiliser un dataset pour l'entraînement avec PyTorch/TensorFlow
```python
# Conversion pour PyTorch
import torch

text = dataset["train"][0]["text"]
label = dataset["train"][0]["label"]
# Utilisation directe pour le DataLoader
```

---

**Documentation officielle** :  
- [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)




### Sources populaires (Hugging Face Hub, Common Crawl)
*(Section à compléter)*




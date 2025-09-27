# Les Datasets

### Table des matières

1. [Définiiton](#définition)
2. [Rôle des Datasets avec les LLM](#role-des-datasets-avec-les-llm)
3. [Types de datasets (pré-entraînement, fine-tuning)](#types-de-datasets-pré-entraînement-fine-tuning)
4. [Qualité des Datasets](#qualité-des-datasets)]
5. [Préparation et nettoyage des données](#préparation-et-nettoyage-des-données)

 
 
9. [Sources populaires (Hugging Face Hub, Common Crawl)](#sources-populaires-hugging-face-hub-common-crawl)

---
### Définition

Un dataset est une collection structurée de données textuelles utilisée pour entraîner, valider ou tester un modèle de langage (composés de textes variés : livres, articles, conversations, pages web, code source, etc.)

---

### Rôle des Datasets avec les LLM

    Entraînement (Training)
        Les LLM apprennent des motifs linguistiques à partir des données textuelles.
        Plus le dataset est large et diversifié, plus le modèle a de chances de généraliser et de comprendre différentes formes d’expression.

    Validation
        On utilise un jeu de données distinct pour vérifier que le modèle n’a pas juste « mémorisé » le dataset d’entraînement.
        Cela permet de détecter l’overfitting (surapprentissage).

    Test
        Un troisième jeu de données, jamais vu par le modèle, permet d’évaluer ses performances « en conditions réelles ».


### Types de datasets (pré-entraînement, fine-tuning)

    - Données ouvertes : Wikipedia, Common Crawl, The Pile, OpenWebText, etc.
    - Données spécialisées : corpus médicaux, juridiques, techniques…
    - Données générées : données synthétiques créées automatiquement pour des tâches spécifiques.

---

### Qualité des Datasets

    - Pertinence : Un dataset de qualité améliore la compréhension du langage et la capacité du modèle à répondre correctement.
    - Biais : Les biais présents dans les données se retrouvent dans le modèle.
    - Sécurité : Datasets mal filtrés peuvent introduire des propos toxiques ou des erreurs.



### Préparation et nettoyage des données

- Préparer les données
- -- Rassembler les données
    Collecte depuis différentes sources : fichiers CSV, bases de données SQL, APIs, web scraping, etc.
    Fusionner les différentes sources si besoin.
- -- Vérifier la structure
    Vérifier que les colonnes attendues sont présentes.
    Uniformiser les formats (dates, nombres, textes…).
- -- Échantillonnage
    Prendre un sous-ensemble si les données sont trop volumineuses.
    Séparer en jeux d’entraînement, validation, et test.

- Nettoyage des données
- -- a) Gestion des valeurs manquantes
    Supprimer les lignes ou colonnes trop incomplètes.
    Imputer les valeurs : remplir avec la moyenne, la médiane, une valeur spécifique, ou par interpolation.
- -- b) Détection et gestion des doublons
    Supprimer les lignes identiques ou similaires.
- -- c) Correction des incohérences
    Uniformiser la casse ("Paris" et "paris" → "Paris").
    Corriger les fautes de frappe, les abréviations non standardisées, etc.
- -- d) Nettoyage du texte (spécifique NLP/LLM)
    Retirer: balises HTML, caractères spéciaux, espaces multiples, emojis ou symboles non désirés.
    Normaliser: accents, ponctuation, casse.
    Tokeniser le texte si besoin.
    Filtrer les langues, retirer les textes trop courts ou non pertinents.
- -- e) Filtrage de contenu
    Éliminer les textes offensants, sensibles ou non adaptés au contexte souhaité.
    Supprimer le spam, les publicités, les parties non textuelles (code, logs, etc. selon le besoin).
- -- f) Conversion des types
    Assurer que les colonnes numériques sont bien en float/int, les dates en datetime, etc.

- Code




---

### Sources populaires (Hugging Face Hub, Common Crawl)
*(Section à compléter)*




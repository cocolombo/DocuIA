Avec LoRA est-ce que ce sont les paramètres des matrices A et B qui "apprennent" ou mis à jour ou si ce sont les paramêtre de la matrice résultate de A * B ? 


# Projet de Fine-Tuning d'un LLM sur un Style de Tweets

Ce projet a pour but de spécialiser un modèle de langage (LLM) pour qu'il apprenne et reproduise un style d'écriture personnel basé sur une archive de tweets. L'objectif est de réaliser un "continued pre-training" (DAPT) pour que le modèle s'imprègne du ton, du vocabulaire et de la cadence spécifiques au corpus de textes.

* [Table des Matières](#table-des-matières)
* [1. Le Processus de Fine-Tuning](#1-le-processus-de-fine-tuning)
    * [Étape 1 : Préparation des Données](#étape-1--préparation-des-données)
    * [Étape 2 : Configuration de l'Environnement](#étape-2--configuration-de-lenvironnement)
    * [Étape 3 : Choix de la Méthode et Chargement du Modèle](#étape-3--choix-de-la-méthode-et-chargement-du-modèle)
    * [Étape 4 : Entraînement (Fine-Tuning)](#étape-4--entraînement-fine-tuning)
    * [Étape 5 : Évaluation et Sauvegarde](#étape-5--évaluation-et-sauvegarde)
    * [Étape 6 : Inférence](#étape-6--inférence)
* [2. Configuration du Projet](#2-configuration-du-projet)
    * [Structure des Données](#structure-des-données)
    * [Chemins des Fichiers](#chemins-des-fichiers)
    * [Structure des Répertoires de Sortie](#structure-des-répertoires-de-sortie)
* [3. Environnement et Outils](#3-environnement-et-outils)
    * [Environnement Technique](#environnement-technique)
    * [Plateformes](#plateformes)
    * [Commandes utiles](#commandes-utiles)
* [4. Exemples de Code et Snippets](#4-exemples-de-code-et-snippets)
    * [Exemple de Fine-Tuning avec Hugging Face](#exemple-de-fine-tuning-avec-hugging-face)
    * [Interprétation des Résultats d'Entraînement](#interprétation-des-résultats-dentraînement)
    * [Trouver le Masque d'un Modèle](#trouver-le-masque-dun-modèle)
* [5. Glossaire des Concepts Clés](#5-glossaire-des-concepts-clés)
    * [Termes Généraux](#termes-généraux)
    * [Techniques de Fine-Tuning](#techniques-de-fine-tuning)
    * [Métriques et Paramètres d'Entraînement](#métriques-et-paramètres-dentraînement)
    * [Bibliothèques et Outils](#bibliothèques-et-outils)
    * [Architectures et Pipelines](#architectures-et-pipelines)
* [6. Prompts](#6-prompts)
    * [Prompt 1 : Guide interactif pour le fine-tuning](#prompt-1--guide-interactif-pour-le-fine-tuning)
    * [Prompt 2 : Refactorisation de code Python](#prompt-2--refactorisation-de-code-python)
* [7. Foire Aux Questions (FAQ)](#7-foire-aux-questions-faq)

## 1. Le Processus de Fine-Tuning

Voici le flux de travail complet, de la donnée brute au modèle spécialisé.

### Étape 1 : Préparation des Données
1.  **Collecte** : Lire les tweets depuis le fichier source (`data/tweets_tous.csv`).
2.  **Nettoyage** : Filtrer les retweets, les réponses automatiques ("Bonjour", etc.) et autres bruits pour ne conserver que le contenu pertinent. Le résultat est sauvegardé dans `data/processed/tweets_tous.csv`.
3.  **Formatage** : Structurer les données nettoyées dans un format compatible avec l'entraînement (ex: JSONL). Chaque entrée doit suivre le template attendu par le modèle.
    * **Format pour Llama 3** : `{"text":"<s>[INST] Instruction [/INST] Réponse</s>"}`
    * **Exemple concret** : `{"text": "<s>[INST] Style: Nimzo | Type: Tweet [/INST] Le PM Legault va finir son terme.</s>"}`
4.  **Division des données** : Répartir le dataset en trois sous-ensembles : entraînement (`train.jsonl`), validation (`validation.jsonl`) et test (`test.jsonl`).
5.  **Tokenisation** : Convertir les textes en séquences de nombres (tokens) à l'aide du tokenizer associé au modèle de base. C'est une étape indispensable avant l'entraînement.

### Étape 2 : Configuration de l'Environnement
* **Infrastructure** : Choisir la plateforme (PC local, Google Colab, Kaggle Notebook).
* **Dépendances** : Installer les bibliothèques Python nécessaires : `transformers`, `datasets`, `torch`, `peft`, `bitsandbytes`, `accelerate`.

### Étape 3 : Choix de la Méthode et Chargement du Modèle
1.  **Charger le Modèle Pré-entraîné** : Charger le modèle de base (ex: `Llama-3-8B`) et son tokenizer depuis un répertoire local ou le Hub Hugging Face.
2.  **Choisir la Méthode de Fine-Tuning** :
    * **Full Fine-Tuning** : Très gourmand, ré-entraîne tous les poids du modèle.
    * **PEFT (Parameter-Efficient Fine-Tuning)** : Approche recommandée pour les ressources limitées.
        * **LoRA/QLoRA** : N'entraîne que de petites matrices ("adaptateurs") ajoutées au modèle, tout en gelant les poids d'origine. QLoRA optimise davantage la mémoire en utilisant la quantification.
3.  **Préparer le Modèle pour PEFT/QLoRA** : Appliquer la configuration pour injecter les adaptateurs LoRA dans le modèle et le préparer à l'entraînement.

### Étape 4 : Entraînement (Fine-Tuning)
1.  **Définir les Hyperparamètres** : Configurer les paramètres de l'entraînement (`TrainingArguments`) comme le taux d'apprentissage (`learning_rate`), le nombre d'époques (`num_train_epochs`), la taille des lots (`per_device_train_batch_size`), etc.
2.  **Initialiser le `Trainer`** : Utiliser la classe `Trainer` de Hugging Face en lui fournissant le modèle, les arguments, et les datasets tokenisés.
3.  **Lancer l'Entraînement** : Exécuter la méthode `trainer.train()`. Pendant cette phase, il est crucial de surveiller la `loss` pour détecter un éventuel surapprentissage (overfitting).

### Étape 5 : Évaluation et Sauvegarde
1.  **Évaluation** : Mesurer la performance du modèle sur l'ensemble de validation et de test à l'aide de la méthode `trainer.evaluate()`.
2.  **Sauvegarde** :
    * Les **checkpoints** sont des sauvegardes intermédiaires automatiques.
    * Le modèle final est sauvegardé avec `trainer.save_model()`. Seuls les adaptateurs PEFT sont sauvegardés, ce qui rend les fichiers très légers.

### Étape 6 : Inférence
* Pour utiliser le modèle, il faut charger le modèle de base, puis y appliquer les adaptateurs fine-tunés. Le modèle est alors prêt à générer du texte dans le style appris.

---

## 2. Configuration du Projet

### Structure des Données

* **Fichiers CSV** :
    * `tweets_tous.csv`: `id_tweet,texte,date_creation`
    * `tweets_tous_nettoyes.csv`: `id_tweet,texte,date_creation`
    * `tweets_formates.csv`: `id_tweet,texte,date_creation,texte_formatte`
* **Statistiques** :
    * Nombre de lignes lues: 161,259
    * Nombre de tweets selon X: 165,300

### Chemins des Fichiers

```
chemin_modele_base        = "./model/llama3-8b/"
chemin_modele_fine_tune   = "./model/llama3-8b-fine-tuned/"
chemin_donnees_train      = "./data/processed/train.jsonl"
chemin_donnees_mini_train = "./data/processed/mini-train.jsonl"
chemin_donnees_validation = "./data/processed/validation.jsonl"
chemin_donnees_test       = "./data/processed/test.jsonl"
```

### Structure des Répertoires de Sortie

Le `Trainer` de Hugging Face crée automatiquement un répertoire de sortie qui contient :
* **Checkpoints** : Des sauvegardes intermédiaires (ex: `checkpoint-500/`, `checkpoint-1000/`) pour reprendre l'entraînement en cas d'interruption.
* **`final_model/`** : Le modèle final prêt à être utilisé pour l'inférence.

```
mon-style-de-tweet-v1/
├── checkpoint-500/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── checkpoint-1000/
│   └── ...
└── final_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```
* `config.json`: Décrit l'architecture du modèle.
* `model.safetensors`: Contient les poids (paramètres) appris.
* `tokenizer.json`, `vocab.json`: Fichiers de configuration du tokenizer.

---

## 3. Environnement et Outils

### Environnement Technique
* **Système d'exploitation** : Linux-Ubuntu 24.04
* **IDE** : PyCharm 2025.2.0.1
* **Langage** : Python 3.12
* **Matériel** : 32Go RAM, 12Go VRAM (GPU)
* **Outils IA** : Ollama, Hugging Face

### Plateformes
* **Local** : Utilise les répertoires `data/processed` et `data/raw`.
* **Google Colab** :
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    # Chemins : drive/MyDrive/colab/twitter2/data/
    ```
* **Kaggle Notebook** :
    ```bash
    !apt-get update && apt-get install -y tree
    !tree -L 3 /kaggle
    ```

### Commandes utiles
* **Synchronisation avec Google Drive via rclone** :
    ```bash
    rclone copy /path/to/local/data GoogleDrive:twitter2/data
    rclone mount GoogleDrive: ~/gdrive --vfs-cache-mode writes
    ```

---

## 4. Exemples de Code et Snippets

### Exemple de Fine-Tuning avec Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 1. Chargement des ressources
model_path = "model/llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Configurer le pad_token s'il est manquant
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_path)

# 2. Chargement et prétraitement des datasets
data_files = {
    "train": "data/processed/train.jsonl",
    "validation": "data/processed/validation.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(preprocess, batched=True)

# 3. Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="model/llama3-8b-finetuned",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
)

# 4. Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()

# 5. Sauvegarde
trainer.save_model("model/llama3-8b-finetuned")
tokenizer.save_pretrained("model/llama3-8b-finetuned")
```

### Interprétation des Résultats d'Entraînement

Voici un exemple de sortie et la signification de chaque métrique :

**Exemple de sortie :**
```
TrainOutput(
    global_step=690,
    training_loss=2.2819,
    metrics={
        'train_runtime': 103.7077,
        'train_samples_per_second': 106.222,
        'train_steps_per_second': 6.653,
        'total_flos': 296779573282176.0,
        'train_loss': 2.2819,
        'epoch': 3.0
    }
)
```
* **global_step** : Nombre total de lots (batches) traités.
* **training_loss / train_loss** : Perte moyenne. Plus elle est basse, mieux le modèle apprend.
* **train_runtime** : Temps total de l'entraînement en secondes.
* **train_samples_per_second** : Nombre d'exemples traités par seconde.
* **train_steps_per_second** : Nombre de lots (steps) traités par seconde.
* **total_flos** : Mesure de la quantité totale de calculs effectués.
* **epoch** : Nombre de fois où l'ensemble du dataset a été parcouru.
* **gradient decent** The direction od steepest increase
* 
### Trouver le Masque d'un Modèle

Le token de masque varie selon les modèles.

```python
from transformers import AutoTokenizer

# Pour CamemBERT
tokenizer_camembert = AutoTokenizer.from_pretrained("camembert-base")
print(f"Masque pour CamemBERT : {tokenizer_camembert.mask_token}") # -> <mask>

# Pour BERT
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Masque pour BERT : {tokenizer_bert.mask_token}") # -> [MASK]
```

---

## 5. Glossaire des Concepts Clés

### Termes Généraux
* **Attention** : Mécanisme permettant au modèle de se concentrer sur des parties spécifiques du texte pour mieux comprendre le contexte.
* **Dataset** : Ensemble de données structurées utilisé pour entraîner ou évaluer un modèle.
* **Embedding** : Représentation vectorielle d'un token ou d'un texte capturant son sens sémantique. `word embedding` est une technique spécifique pour représenter les mots sous forme de vecteurs.
* **Inference** : Utilisation d'un modèle entraîné pour générer des prédictions.
* **Mixture of Experts (MoE)** : Architecture où plusieurs sous-réseaux ("experts") sont activés sélectivement par un routeur pour traiter chaque token.
* **Overfitting** : Situation où un modèle apprend "par cœur" les données d'entraînement et perd sa capacité à généraliser sur de nouvelles données.
* **Perplexité** : Mesure de l'incertitude d'un modèle pour prédire le prochain token. Une faible perplexité indique un meilleur modèle.
* **Pré-entraînement** : Phase initiale d'apprentissage d'un LLM sur un très large corpus de textes.
* **Token** : Unité de base du texte (mot, sous-mot ou caractère) que le modèle traite.
* **Tokenisation** : Processus de segmentation d'un texte en tokens.
* **Zero-shot learning** : Capacité d'un modèle à effectuer une tâche pour laquelle il n'a pas été explicitement entraîné.
* **TQDM** bibliothèque Python utilisée pour afficher une barre de progression lors de l’exécution de boucles

### Techniques de Fine-Tuning
* **Fine-tuning** : Ajustement d'un modèle pré-entraîné sur des données spécifiques pour l'adapter à une tâche ciblée.
* **LoRA (Low-Rank Adaptation)** : Technique de PEFT qui gèle le modèle original et n'entraîne que de petites matrices ("adaptateurs") ajoutées à certaines couches, réduisant considérablement le coût de calcul.
* **The rank of a matrix** is the maximum number of linearly independent rows or, equivalently, columns in the matrix. Le rang de la matrice LoRA, noté r, est un hyperparamètre que vous définissez avant de lancer l'entraînement, au moment de la configuration du modèle. Le rang de la matrice LoRA, noté r, est un hyperparamètre que vous définissez avant de lancer l'entraînement, au moment de la configuration du modèle. Le rang d'une matrice est un concept clé en algèbre linéaire qui mesure, intuitivement, le "niveau d'indépendance" ou la "quantité d'information utile" contenue dans la matrice.
* **low-rank matrix** is a matrix with a rank significantly lower than its number of rows or columns, meaning its rows (or columns) are not all linearly independent. 
* **PEFT (Parameter-Efficient Fine-Tuning)** : Famille de méthodes (dont LoRA est un exemple) visant à n'ajuster qu'une petite fraction des paramètres du modèle.
* **Quantisation** : Technique de réduction de la précision des poids du modèle (ex: de 32 bits à 4 bits) pour réduire la consommation de mémoire et accélérer l'inférence. On change le format (souvent flottant → entier)
* **Réduction de Précision** : Technique similaire à la quantisation. On garde le format flottant, mais avec moins de bits.
* **QLoRA (Quantized LoRA)** : Combine la quantification et LoRA pour un fine-tuning encore plus efficace en termes de mémoire.
* **DAPT** (Domain-adaptive Pre-Training) : Technique qui pré-entraîne davantage les LLM sur des données non étiquetées spécifiques à un domaine à l'aide de la modélisation linguistique masquée (MLM) afin d'améliorer la spécialisation.


### Métriques et Paramètres d'Entraînement
* **Epoch** : Un passage complet à travers l'ensemble des données d'entraînement.
* **grad_norm (norme du gradient)** : Mesure l'ampleur des mises à jour des poids du modèle à chaque étape.
* **Hyperparamètres** : Paramètres définis avant l'entraînement qui contrôlent le processus (ex: taux d'apprentissage, taille de lot).
* **logging_steps** : Fréquence (en nombre de steps) à laquelle afficher les métriques de progression.
* **loss** : Mesure de l'erreur entre les prédictions du modèle et les véritables données. C'est la valeur que l'on cherche à minimiser.
* **save_steps** : Fréquence (en nombre de steps) à laquelle sauvegarder un checkpoint du modèle.
* **split="train"** : Argument utilisé dans `load_dataset` pour sélectionner une partie spécifique (ici, l'ensemble d'entraînement) d'un jeu de données.
* **steps** : Nombre total d'itérations d'entraînement. Calcul : `(Taille du dataset / Taille de lot) * Nombre d'époques`.

### Bibliothèques et Outils
* **accelerate** : Bibliothèque Hugging Face qui simplifie la gestion du matériel (GPU/CPU/multi-GPU).
* **bitsandbytes** : Bibliothèque permettant la quantification des modèles (essentielle pour QLoRA).
* **Unsloth** : Bibliothèque optimisée qui accélère le fine-tuning des LLMs (comme Llama, Mistral) en réduisant drastiquement la consommation de VRAM.

### Architectures et Pipelines
* **BERT** : Une architecture de modèle de type "Encoder-only". `bert-base-cased` est un checkpoint spécifique (un ensemble de poids) pour cette architecture.
* **Encoder / Decoder / Encoder-decoder** :
    * **Encoder only** : Transforme le texte d'entrée en une représentation numérique (features). À chaque étape, les couches d'attention peuvent accéder à tous les mots de la phrase d'entrée. Bi-directional attention: BERT, DistilBERT, ModernBERT
    * **Decoder only** : Utilise cette représentation pour générer le texte de sortie. Unidirectionel (acccès au cobtexte de gauche seulement). GPT-2, GPT-Neo, Llama, Gemma, Deepseek-v3. Tâches: Génération de texte, tésumé, traduction, Question-Réponde, Code, Raisonnement, "Few-shot learning" 
    * **Encoder-Decoder** Sequence to sequence. Prediction de la suite du texte. BART, T5, mBART, Marian, 

* **Autoregressive models** génère chaque nouveau token en se basant sur tous les tokens précédents de la séquence. À chaque étape, le modèle prend la séquence déjà générée prédit le prochain token le plus probable. La prédiction se poursuit jusqu’à la fin de la séquence (ou jusqu’à un token de fin).
* **Pipelines** : Classes de haut niveau de Hugging Face pour utiliser facilement des modèles pour des tâches courantes (traduction, résumé, génération de texte, etc.). Le pipeline sert à exécuter un modèle déjà entraîné, pas à l’entraîner. C'est principalement un outil d'inférence

---

## 6. Prompts

### Prompt 1 : Guide interactif pour le fine-tuning

* **Titre** : Guide interactif pour fine-tuner Llama 3 avec mes archives de tweets en utilisant Ollama.
* **Objectif** : Être guidé pas à pas par un expert IA pour fine-tuner Llama 3 sur mon style d'écriture en utilisant Ollama et un échantillon de 100 tweets pour commencer.
* **Méthodologie** : Collaboration interactive où chaque étape est présentée et validée avant de passer à la suivante, avec du code Python commenté et modulaire.
* **Plan suggéré** :
    1.  **Préparation des Données** : Charger et formater un échantillon en JSONL.
    2.  **Configuration** : Créer le `Modelfile` pour Ollama et recommander des paramètres.
    3.  **Lancement** : Fournir la commande `ollama create` exacte.
    4.  **Test** : Montrer comment exécuter le modèle final avec `ollama run`.

### Prompt 2 : Refactorisation de code Python

* **Rôle** : Développeur Python expérimenté expert en sécurité, performance et refactorisation.
* **Objectif** : Examiner et améliorer un code Python pour qu'il respecte les plus hauts standards de qualité.
* **Instructions** :
    * Analyser le contexte pour réutiliser le code existant.
    * Effectuer un audit de sécurité.
    * Supprimer les redondances et le code mort (TODOs).
    * Remplacer les valeurs "magiques" par des constantes.
    * Optimiser les performances et les algorithmes.
    * Respecter les bonnes pratiques (lisibilité, gestion d'erreurs, modularité).
* **Format de sortie** : Code final refactorisé avec un résumé des modifications apportées.

---

## 7. Foire Aux Questions (FAQ)

* **Pourquoi `split->train` dans `test/dataset_info.json` ?**
    * [Lien de discussion](https://github.com/copilot/share/020851b8-4ba0-8094-8011-140140a04999)
* **Pourquoi un chemin contient-il `./` ?**
    * [Lien de discussion](https://github.com/copilot/share/c20113b8-0284-8832-9941-860860684988)
* **Pourquoi ne peut-on pas simplement copier les fichiers d'Ollama ?**
    * [Lien de discussion](https://gemini.google.com/app/0c348f71e2349304)
* **Qu'est-ce que le "préchauffage" du taux d'apprentissage (`warmup_steps`) ?**
    * [Lien de discussion](https://gemini.google.com/app/745a33b423c7a4b1)
* **Les versions de Llama3 (Meta vs Ollama) sont-elles identiques ?**
    * [Lien de discussion](https://grok.com/share/c2hhcmQtMg%3D%3D_f3f103b1-4250-4fc7-86e5-2fd09f9f34ac)
* **Ressources sur la tokenisation :**
    * [Lien 1](https://gemini.google.com/app/018aa5e35a1b5146)
    * [Lien 2](https://grok.com/share/c2hhcmQtMg%3D%3D_ee4742d2-f4a0-4659-92be-b45a497c159d)
* **Quels modèles LLM de Google peuvent être fine-tunés ?**
    * [Gemma](https://github.com/copilot/share/4261422a-0ba4-8c12-b003-9641442249da)
* **Fine-tuning vs pré-entraînement continu :**
    * [Lien de discussion](https://gemini.google.com/app/ea602cce5c2d9510)


#### Historique des modèles 


GPT, the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results [June 2018]

BERT, another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!) [October 2018]

GPT-2, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns [February 2019]

T5, A multi-task focused implementation of the sequence-to-sequence Transformer architecture. [October 2019]

GPT-3, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning) [May 2020]

InstructGPT, a version of GPT-3 that was trained to follow instructions better This list is far from comprehensive, and is just meant to highlight a few of the different kinds of Transformer models. Broadly, they can be grouped into three categories: [January 2022]

Llama, a large language model that is able to generate text in a variety of languages. [January 2023]

Mistral, a 7-billion-parameter language model that outperforms Llama 2 13B across all evaluated benchmarks, leveraging grouped-query attention for faster inference and sliding window attention to handle sequences of arbitrary length. [March 2023]

Gemma 2, a family of lightweight, state-of-the-art open models ranging from 2B to 27B parameters that incorporate interleaved local-global attentions and group-query attention, with smaller models trained using knowledge distillation to deliver performance competitive with models 2-3 times larger. [May 2024]

SmolLM2, a state-of-the-art small language model (135 million to 1.7 billion parameters) that achieves impressive performance despite its compact size, and unlocking new possibilities for mobile and edge devices. [November 2024]

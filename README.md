
1.  [Introduction aux LLM](./00_Introduction_LLM.md)
2.  [L'architecture Transformers](./01_Architecture_Transformers.md)
3.  [La Tokenisation](./02_Tokenisation.md)
4.  [Les Datasets](./03_Datasets.md)
5.  [Entraînement et Fine-Tuning](./04_Entrainement_et_Fine_Tuning.md)
6.  [Inférence et Quantization](./05_Inference_et_Quantization.md)
7.  [Évaluation des Modèles](./06_Evaluation_des_Modeles.md)
8.  [Éthique et Limites](./07_Ethique_et_Limites.md)
9.  [Glossaire](./08_Glossaire.md)

Avec LoRA est-ce que ce sont les paramètres des matrices A et B qui "apprennent" ou mis à jour ou si ce sont les paramêtre de la matrice résultate de A * B ? 

# 1. Le Processus de Fine-Tuning

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
        xxx
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



* `config.json`: Décrit l'architecture du modèle.
* `model.safetensors`: Contient les poids (paramètres) appris.
* `tokenizer.json`, `vocab.json`: Fichiers de configuration du tokenizer.

---


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


### Métriques et Paramètres d'Entraînement
* **grad_norm (norme du gradient)** : Mesure l'ampleur des mises à jour des poids du modèle à chaque étape.
* **logging_steps** : Fréquence (en nombre de steps) à laquelle afficher les métriques de progression.
* **save_steps** : Fréquence (en nombre de steps) à laquelle sauvegarder un checkpoint du modèle.
* **split="train"** : Argument utilisé dans `load_dataset` pour sélectionner une partie spécifique (ici, l'ensemble d'entraînement) d'un jeu de données.
* **steps** : Nombre total d'itérations d'entraînement. Calcul : `(Taille du dataset / Taille de lot) * Nombre d'époques`.

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

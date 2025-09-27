# Glossaire

### Table des matières
-   A-Z (liste alphabétique des termes techniques)


Voici une liste alphabétique de termes techniques essentiels liés aux LLM. Elle couvre les concepts fondamentaux de l'architecture, de l'entraînement et de l'utilisation des modèles de langage.

### A

* **accelerate** : Bibliothèque Hugging Face qui simplifie la gestion du matériel (GPU/CPU/multi-GPU).
* **API (Application Programming Interface)** : Interface de programmation qui permet à différentes applications de communiquer entre elles. Les LLM sont souvent accessibles via des API.
* **Apprentissage par renforcement (Reinforcement Learning - RL)** : Domaine du machine learning où un agent apprend à prendre des décisions en effectuant des actions dans un environnement pour maximiser une récompense. Le RLHF (Reinforcement Learning from Human Feedback) est une technique utilisée pour affiner les LLM.
* **Architecture (Model Architecture)** : La conception et la structure d'un modèle de langage, comme l'architecture Transformer.
* **Attention (Attention Mechanism)** : Mécanisme qui permet à un modèle de se concentrer sur les parties les plus pertinentes de l'input pour mieux comprendre le contexte lors de la génération de l'output.
* **Autoregressive models** génère chaque nouveau token en se basant sur tous les tokens précédents de la séquence. À chaque étape, le modèle prend la séquence déjà générée prédit le prochain token le plus probable. La prédiction se poursuit jusqu’à la fin de la séquence (ou jusqu’à un token de fin).
* 
### B

* **Biais (Bias)** : Tendance d'un modèle à produire des résultats systématiquement erronés en raison de données d'entraînement non représentatives.
* **BERT** : Une architecture de modèle de type "Encoder-only". `bert-base-cased` est un checkpoint spécifique (un ensemble de poids) pour cette architecture.
* **bitsandbytes** : Bibliothèque permettant la quantification des modèles (essentielle pour QLoRA).

* **BLEU (Bilingual Evaluation Understudy)** : Métrique utilisée pour évaluer la qualité d'un texte généré par une machine.

### C

* **Contexte (Context Window)** : La quantité de texte qu'un modèle peut prendre en compte à un moment donné.
* **Corpus** : Un grand ensemble de textes utilisé pour l'entraînement des modèles de langage.

### D 

* **DAPT** (Domain-adaptive Pre-Training) : Technique qui pré-entraîne davantage les LLM sur des données non étiquetées spécifiques à un domaine à l'aide de la modélisation linguistique masquée (MLM) afin d'améliorer la spécialisation.
* **Dataset** : Ensemble de données utilisé pour l'entraînement, la validation ou le test d'un modèle.
* **Deep Learning** : Sous-domaine du machine learning basé sur des réseaux de neurones artificiels avec plusieurs couches.
* **Déploiement (Deployment)** : Processus de mise en production d'un modèle pour qu'il puisse être utilisé par les utilisateurs finaux.

### E

* **Embedding** : Représentation vectorielle d'un token ou d'un texte capturant son sens sémantique. `word embedding` est une technique spécifique pour représenter les mots sous forme de vecteurs.
* **Encodeur (Encoder)** : Partie d'un modèle Transformer qui traite le texte d'entrée.
* **Encoder / Decoder / Encoder-decoder** :
  **Encoder only** : Transforme le texte d'entrée en une représentation numérique (features). À chaque étape, les couches d'attention peuvent accéder à tous les mots de la phrase d'entrée. Bi-directional attention: BERT, DistilBERT, ModernBERT
  **Decoder only** : Utilise cette représentation pour générer le texte de sortie. Unidirectionel (acccès au cobtexte de gauche seulement). GPT-2, GPT-Neo, Llama, Gemma, Deepseek-v3. Tâches: Génération de texte, tésumé, traduction, Question-Réponde, Code, Raisonnement, "Few-shot learning" 
  **Encoder-Decoder** Sequence to sequence. Prediction de la suite du texte. BART, T5, mBART, Marian, 
  Voir **BERT** 
* **Entraînement (Training)** : Processus d'apprentissage d'un modèle à partir d'un ensemble de données.
* **Epoch** : Un passage complet à travers l'ensemble des données d'entraînement.
* **Éthique de l'IA (AI Ethics)** : Domaine qui étudie les implications morales et sociales de l'intelligence artificielle.
* **Évaluation (Evaluation)** : Processus de mesure de la performance d'un modèle.

### F

* **Fine-tuning** : Processus d'adaptation d'un modèle pré-entraîné à une tâche spécifique en le ré-entraînant sur un ensemble de données plus petit et plus spécifique pour l'adapter à une tâche ciblée.


### G

* **Génération de texte (Text Generation)** : Processus de création de texte par un modèle de langage.
* **GPT (Generative Pre-trained Transformer)** : Famille de modèles de langage développée par OpenAI.

### H

* **Hallucination** : Phénomène où un LLM génère des informations fausses ou trompeuses.
* **Hyperparamètres (Hyperparameters)** : Paramètres qui ne sont pas appris par le modèle mais qui sont définis avant l'entraînement. (Ex: taux d'apprentissage, taille de lot).

### I

* **IA Générative (Generative AI)** : Type d'IA qui peut créer de nouveaux contenus, comme du texte, des images ou de la musique.
* **Inférence (Inference)** : Processus d'utilisation d'un modèle entraîné pour faire des prédictions sur de nouvelles données.

### L

* **LLM (Large Language Model)** : Grand modèle de langage, un type de modèle d'IA entraîné sur d'énormes quantités de texte.
* **LoRA (Low-Rank Adaptation)** : Technique de fine-tuning efficace qui réduit le nombre de paramètres à entraîner.
* **QLoRA (Quantized LoRA)** : Combine la quantification et LoRA pour un fine-tuning encore plus efficace en termes de mémoire.
* **LoRA (Low-Rank Adaptation)** : Technique de PEFT qui gèle le modèle original et n'entraîne que de petites matrices ("adaptateurs") ajoutées à certaines couches, réduisant considérablement le coût de calcul.
* 
* **loss** : Mesure de l'erreur entre les prédictions du modèle et les véritables données. C'est la valeur que l'on cherche à minimiser.

### M

* **Machine Learning** : Domaine de l'IA qui se concentre sur le développement d'algorithmes qui permettent aux ordinateurs d'apprendre à partir de données.
* **Le rang d'une matrice** est le nombre maximal de lignes ou, de manière équivalente, de colonnes linéairement indépendantes dans la matrice. 
Le rang de la matrice LoRA, noté r, est un hyperparamètre que vous définissez avant de lancer l'entraînement, au moment de la configuration du modèle. Le rang de la matrice LoRA, noté r, est un hyperparamètre que vous définissez avant de lancer l'entraînement, au moment de la configuration du modèle. Le rang d'une matrice est un concept clé en algèbre linéaire qui mesure, intuitivement, le "niveau d'indépendance" ou la "quantité d'information utile" contenue dans la matrice.
* **Une matrice de rang faible** est une matrice dont le rang est nettement inférieur à son nombre de lignes ou de colonnes, ce qui signifie que ses lignes (ou colonnes) ne sont pas toutes linéairement indépendantes.
* **Métriques (Metrics)** : Mesures utilisées pour évaluer la performance d'un modèle, comme la perplexité ou le score BLEU.
* **Mixture of Experts (MoE)** : Architecture où plusieurs sous-réseaux ("experts") sont activés sélectivement par un routeur pour traiter chaque token.
* **Modèle de langage (Language Model)** : Modèle statistique qui apprend les probabilités de séquences de mots dans une langue.

### N

* **NLP (Natural Language Processing)** : Traitement du langage naturel, domaine de l'IA qui se concentre sur l'interaction entre les ordinateurs et le langage humain.
* **Neurone (Neuron)** : Unité de base d'un réseau de neurones.

### O

* **Overfitting** : Situation où un modèle apprend "par cœur" les données d'entraînement et perd sa capacité à généraliser sur de nouvelles données.

### P

* **Paramètres (Parameters)** : Les poids et les biais d'un modèle qui sont appris pendant l'entraînement.
* **PEFT (Parameter-Efficient Fine-Tuning)** : Famille de méthodes (dont LoRA est un exemple) visant à n'ajuster qu'une petite fraction des paramètres du modèle.
* **Perplexité** : Mesure de l'incertitude d'un modèle pour prédire le prochain token. Une faible perplexité indique un meilleur modèle.
* **Pipelines** : Classes de haut niveau de Hugging Face pour utiliser facilement des modèles pour des tâches courantes (traduction, résumé, génération de texte, etc.). Le pipeline sert à exécuter un modèle déjà entraîné, pas à l’entraîner. C'est principalement un outil d'inférence
* **Réduction de Précision** : Technique similaire à la quantisation. On garde le format flottant, mais avec moins de bits.
* **Pré-entraînement (Pre-training)** : Entraînement initial d'un modèle sur un grand corpus de texte non étiqueté.
* **Prompt** : L'instruction ou la question donnée à un LLM pour générer une réponse.

### Q

* **Quantization** : Technique de réduction de la précision des poids du modèle (ex: de 32 bits à 4 bits) pour réduire la consommation de mémoire et accélérer l'inférence. On change le format (souvent flottant → entier)

### R

* **Réseau de neurones (Neural Network)** : Modèle de calcul inspiré du cerveau humain, utilisé dans le deep learning.
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** : Métrique utilisée pour évaluer la qualité des résumés de texte.

### S

* **Sentiment Analysis** : Analyse des sentiments, processus de détermination du ton émotionnel d'un texte.
* **Summarization** : Résumé de texte, processus de création d'une version plus courte d'un texte tout en conservant les informations les plus importantes.

### T

* **Température (Temperature)** : Hyperparamètre qui contrôle le degré d'aléatoire dans la génération de texte.
* **Token** : Unité de base du texte pour un modèle de langage, qui peut être un mot, un sous-mot ou un caractère.
* **Tokenisation (Tokenization)** : Processus de découpage du texte en tokens.
* **TQDM** bibliothèque Python utilisée pour afficher une barre de progression lors de l’exécution de boucles
* **Transformer** : Architecture de réseau de neurones qui est devenue la norme pour les LLM.

### U

* **Unsloth** : Bibliothèque optimisée qui accélère le fine-tuning des LLMs (comme Llama, Mistral) en réduisant drastiquement la consommation de VRAM.

### Z

* **Zero-shot learning** : Capacité d'un modèle à effectuer une tâche sans avoir été explicitement entraîné pour celle-ci. Au moment du test, on présente au modèle des échantillons provenant de classes qui n'ont pas été observées pendant l'entraînement et doit prédire la classe à laquelle ils appartiennent.

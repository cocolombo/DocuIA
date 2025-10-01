# La Tokenisation

### Table des matières

1. [Pourquoi tokeniser le texte ?](#pourquoi-tokeniser-le-texte-)
2. [Stratégies de tokenisation (Word, Subword)](#stratégies-de-tokenisation-word-subword)
3. [Exemples d'algorithmes (BPE, WordPiece)](#exemples-dalgorithmes-bpe-wordpiece)
4. [Tokens spéciaux (PAD, CLS, SEP)](#tokens-spéciaux-pad-cls-sep)
5. [Applications et limites](#applications-et-limites)
6. [Outils et bibliothèques](#outils-et-bibliothèques)
7. [Exemples de tokenisation](#exemples-de-tokenisation)

---

### Pourquoi tokeniser le texte ?
La tokenisation consiste à découper un texte en unités élémentaires appelées *tokens* (mots, sous-mots, caractères, etc.).  
C'est une étape fondamentale dans le traitement automatique du langage (NLP) parce qu'elle permet de transformer un texte brut en une séquence manipulable par des modèles informatiques.  
**Objectifs principaux :**
- Séparer les mots, ponctuations, et autres entités pour faciliter leur analyse.
- Réduire la complexité du texte et standardiser les entrées pour les modèles.
- Permettre la gestion des mots inconnus ou rares.

---

### Stratégies de tokenisation (Word, Subword)
La stratégie de tokenisation dépend des besoins et du modèle utilisé :  
- **Word Tokenization** : chaque mot du texte devient un token. Simple à implémenter, mais ne gère pas les mots inconnus (OOV, Out-Of-Vocabulary).
- **Character Tokenization** : chaque caractère est un token. Permet de gérer tous les mots, mais augmente la longueur des séquences.
- **Subword Tokenization** : découpe les mots en unités plus petites (morphèmes, syllabes, ou fragments fréquents). C’est le compromis utilisé par la plupart des modèles modernes car il gère mieux la diversité linguistique et les mots rares.

---

### Exemples d'algorithmes (BPE, WordPiece)
**BPE (Byte-Pair Encoding)**  
- Algorithme de compression adapté à la tokenisation.
- Fusionne les paires de caractères ou de sous-mots les plus fréquentes.
- Permet de créer un vocabulaire compact qui couvre la majorité des mots tout en gérant les mots inconnus par décomposition.

**WordPiece**  
- Utilisé par les modèles comme BERT.
- Similaire à BPE mais les fusions sont basées sur la probabilité d’apparition et la maximisation de la vraisemblance du corpus.
- Produit des tokens qui commencent souvent par "##" pour indiquer qu’ils ne sont pas des débuts de mots.

**SentencePiece**  
- Peut fonctionner sans espaces (utile pour les langues comme le japonais).
- Génère des sous-mots ou des caractères selon la fréquence.

---

### Tokens spéciaux (PAD, CLS, SEP)
Les modèles modernes intègrent des *tokens spéciaux* pour structurer les séquences :
- **[PAD]** : utilisé pour compléter les séquences à une longueur fixe lors du batch processing.
- **[CLS]** : token ajouté au début pour classifier la séquence (utilisé par BERT).
- **[SEP]** : sépare différentes parties d’une séquence (par exemple, question/réponse ou phrase 1/phrase 2).
- D’autres tokens existent selon les modèles : [MASK], [UNK], etc.

---

### Applications et limites
**Applications :**
- Préparation des données pour l’entraînement de modèles de NLP (classification, traduction, génération, etc.).
- Indexation et recherche de texte.
- Analyse de sentiments, extraction d’information.

**Limites :**
- Les langues morphologiquement complexes ou sans espaces posent des défis.
- Les erreurs de tokenisation peuvent impacter les performances du modèle.
- La tokenisation peut perdre le sens ou la structure grammaticale.

---

### Outils et bibliothèques
- **NLTK** : offre des fonctions de tokenisation de mots et de phrases (Python).
- **spaCy** : tokenisation rapide et précise, prise en charge de plusieurs langues.
- **Hugging Face Tokenizers** : très performant, compatible avec les modèles Transformers.
- **SentencePiece** : tokenisation basée sur les sous-mots, supporte Unicode.
- **BERT Tokenizer** : implémentation WordPiece spécialement pour BERT.

---

### Exemples de tokenisation

#### 1. Tokenisation par mot
Texte :  
`Le chat mange.`  
Tokens :  
`["Le", "chat", "mange", "."]`

#### 2. Tokenisation par caractère
Texte :  
`chat`  
Tokens :  
`["c", "h", "a", "t"]`

#### 3. Tokenisation subword (WordPiece/BPE)
Texte :  
`incroyablement`  
Tokens (WordPiece) :  
`["in", "##croy", "##able", "##ment"]`  
Tokens (BPE/SentencePiece, exemple) :  
`["incroy", "able", "ment"]`

#### 4. Ajout de tokens spéciaux  
Texte :  
`Le chat mange.`  
Tokens (BERT) :  
`["[CLS]", "Le", "chat", "mange", ".", "[SEP]"]`

#### 5. Gestion d’un mot inconnu  
Texte :  
`pythonicité` (inconnu du vocabulaire)  
Tokens (WordPiece) :  
`["python", "##ic", "##ité"]`  
Tokens (si aucun fragment n’existe) :  
`["[UNK]"]`

Si le vocabulaire contient les fragments "python" et "##icité", alors le mot pythonicité sera tokenisé en:
["python", "##icité"]

Si le vocabulaire possède les fragments "python", "##ic", et "##ité", la tokenisation sera:
["python", "##ic", "##ité"]

Si le vocabulaire ne contient aucun fragment approprié après "python", le reste du mot pourrait être remplacé par [UNK].

---
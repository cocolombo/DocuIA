# Entraînement et Fine-Tuning

https://grok.com/share/c2hhcmQtMg%3D%3D_034ef7a8-49c5-4f72-a5cd-eb1428606fae

https://g.co/gemini/share/b980640f6436

### Table des matières

1. Pré-entraînement (Pre-training)
2. Spécialisation : le Fine-Tuning
3. Techniques de Fine-Tuning efficaces (LoRA, QLoRA)
4. Apprentissage par renforcement (RLHF)

Les matrices low-rank A et B détails:
[](https://grok.com/share/c2hhcmQtMg%3D%3D_034ef7a8-49c5-4f72-a5cd-eb1428606fae)
[](https://g.co/gemini/share/b980640f6436)

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

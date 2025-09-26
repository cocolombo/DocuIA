# Entraînement et Fine-Tuning

### Table des matières
1.  Pré-entraînement (Pre-training)
2.  Spécialisation : le Fine-Tuning
3.  Techniques de Fine-Tuning efficaces (LoRA, QLoRA)
4.  Apprentissage par renforcement (RLHF)

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
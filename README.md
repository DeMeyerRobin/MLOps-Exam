---
title: GOT House Predictor
emoji: 🐉
colorFrom: blue
colorTo: red
sdk: docker
app_file: Fastapi/app.py
pinned: false
---

# Game of Thrones House Predictor

Predicts which Game of Thrones house a character belongs to based on their traits using a Decision Tree classifier.

## API Usage

This is a FastAPI endpoint. Access the interactive docs at `/docs`

### POST `/predict`

Send a JSON payload with character attributes:
```json
{
  "region": "The North",
  "primary_role": "Commander",
  "alignment": "Lawful Good",
  "status": "Alive",
  "species": "Human",
  "honour_1to5": 4,
  "ruthlessness_1to5": 2,
  "intelligence_1to5": 3,
  "combat_skill_1to5": 4,
  "diplomacy_1to5": 3,
  "leadership_1to5": 4,
  "trait_loyal": true,
  "trait_scheming": false
}
```

Returns:
```json
{
  "predicted_house": "Stark",
  "input": { ... }
}
```

## Model

Trained on Game of Thrones character dataset using scikit-learn Decision Tree Classifier with Azure ML.
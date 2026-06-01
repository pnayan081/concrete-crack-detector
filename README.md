# Concrete Crack Detection API

FastAPI backend for classifying concrete images as cracked or non-cracked.
## 0. Virtual environment creation

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 1. Install dependencies

```powershell
pip install -r requirements.txt
```

## 2. Train and export the model

Use folders that contain the original images for each class.

```powershell
python train_export_model.py `
  --non-cracked-dir "dataset/Negative" `
  --cracked-dir "dataset/Positive"
```

This creates:

- `models/concrete_crack.keras`
- `models/metadata.json`

The label mapping is fixed as:

- `0 = Non-Cracked`
- `1 = Cracked`

## 3. Start the API

```powershell
uvicorn api.main:app --reload
```

Open the API docs at `http://127.0.0.1:8000/docs`.

## 4. Predict

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@sample.jpg"
```

Response:

```json
{
  "filename": "sample.jpg",
  "label": "Cracked",
  "has_crack": true,
  "confidence": 0.94,
  "crack_probability": 0.94,
  "threshold": 0.5,
  "image": {
    "mime_type": "image/jpeg",
    "base64": "..."
  }
}
```

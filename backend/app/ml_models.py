from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "scibert_domain_classifier")
)
class PaperClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        with open(os.path.join(model_dir, "label_map.json")) as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def classify(self, text: str) -> dict:
        inputs = self.tokenizer(
            text[:2000], return_tensors="pt", truncation=True,
            padding=True, max_length=256
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = probs.argmax().item()

        top3_ids = probs.argsort(descending=True)[:3]
        top3 = [
            {"domain": self.label_map[i.item()], "confidence": round(probs[i].item(), 3)}
            for i in top3_ids
        ]
        return {
            "predicted_domain": self.label_map[pred_id],
            "confidence": round(probs[pred_id].item(), 3),
            "top3": top3
        }
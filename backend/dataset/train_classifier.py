import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
import json

MODEL_NAME = "allenai/scibert_scivocab_uncased"
OUTPUT_DIR = "backend/models/scibert_domain_classifier"

df = pd.read_csv("backend/dataset/domain_dataset.csv")

encoder = LabelEncoder()
df["label_id"] = encoder.fit_transform(df["label"])
label_names = list(encoder.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train(
    resume_from_checkpoint="backend/models/scibert_domain_classifier/checkpoint-150"
)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/label_map.json", "w") as f:
    json.dump({i: name for i, name in enumerate(label_names)}, f)

print("Saved fine-tuned model to", OUTPUT_DIR)
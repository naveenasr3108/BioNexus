import os
import fitz
from sklearn.metrics import accuracy_score, classification_report

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", "app")
    )
)

from ml_models import PaperClassifier


# Folder containing test papers
PAPERS_DIR = os.path.join(
    os.path.dirname(__file__),
    "papers"
)

classifier = PaperClassifier()

y_true = []
y_pred = []


for domain in os.listdir(PAPERS_DIR):

    domain_path = os.path.join(PAPERS_DIR, domain)

    if not os.path.isdir(domain_path):
        continue

    for filename in os.listdir(domain_path):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(domain_path, filename)

        # Extract PDF text
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        # Run existing SciBERT classifier
        result = classifier.classify(text)

        predicted = result["predicted_domain"]

        y_true.append(domain.title())
        y_pred.append(predicted)

        print(f"\nFile: {filename}")
        print(f"Actual:    {domain}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {result['confidence']}")


# Calculate metrics
print("\n" + "=" * 50)
print("SCIBERT CLASSIFIER EVALUATION")
print("=" * 50)

print(f"\nNumber of papers: {len(y_true)}")

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)
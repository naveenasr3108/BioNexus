from sentence_transformers import CrossEncoder
import numpy as np


class ContradictionDetector:

    def __init__(self):
        print("Loading DeBERTa NLI model...")
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        print("DeBERTa NLI model loaded")

    def compare(self, text1, text2):

        scores = self.model.predict([(text1, text2)])[0]

        probabilities = np.exp(scores) / np.sum(np.exp(scores))

        labels = ["contradiction", "entailment", "neutral"]
        index = probabilities.argmax()

        return {
            "label": labels[index],
            "score": round(float(probabilities[index]), 3)
        }
from Bio import Entrez
import pandas as pd
import time

Entrez.email = "naveenasr2006@gmail.com"  

DOMAINS = {
    "Oncology": "cancer treatment therapy",
    "Cardiology": "cardiovascular heart disease",
    "Neurology": "neurological brain disorder",
    "Immunology": "immune system immunotherapy",
    "Infectious Disease": "infectious disease pathogen",
    "Genetics": "genetic mutation genome",
    "Public Health": "public health epidemiology",
    "Pharmacology": "drug pharmacokinetics pharmacology",
}

def fetch_abstracts(query, max_results=70):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    ids = Entrez.read(handle)["IdList"]
    handle.close()

    if not ids:
        return []

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    for article in records.get("PubmedArticle", []):
        try:
            abstract_parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            text = " ".join(str(p) for p in abstract_parts)
            if len(text.split()) > 30:  # skip near-empty abstracts
                abstracts.append(text)
        except (KeyError, TypeError):
            continue
    return abstracts

rows = []
for domain, query in DOMAINS.items():
    print(f"Fetching: {domain}")
    abstracts = fetch_abstracts(query)
    for text in abstracts:
        rows.append({"text": text, "label": domain})
    time.sleep(0.5)  # respect NCBI rate limits

df = pd.DataFrame(rows)
df.to_csv("backend/dataset/domain_dataset.csv", index=False)
print(df["label"].value_counts())
print(f"Total samples: {len(df)}")
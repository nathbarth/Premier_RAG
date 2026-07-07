import csv

CORPUS_CSV_PATH = "data/corpus_rag.csv"


def load_corpus(csv_path: str = CORPUS_CSV_PATH) -> list[dict]:
    chunks = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append({
                "id": row["id"],
                "text": row["text"],
                "source": row["source"],
                "categorie": row["categorie"],
            })
    return chunks
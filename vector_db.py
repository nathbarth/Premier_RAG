"""
Brique 1 - La base vectorielle persistante.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config


class VectorDB:
    def __init__(self, db_path=config.CHROMA_DB_PATH,
                 collection_name=config.CHROMA_COLLECTION_NAME,
                 chunks=None):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)

        collection_exists = collection_name in [
            c.name for c in self.client.list_collections()
        ]

        if collection_exists:
            self._reload_existing_collection()
        elif chunks:
            self._create_collection(chunks)
        else:
            raise ValueError(
                f"Aucune base trouvee a '{db_path}' pour la collection "
                f"'{collection_name}', et aucun chunk fourni pour en creer une."
            )

    def _create_collection(self, chunks):
        print(f"[VectorDB] Creation d'une nouvelle collection '{self.collection_name}'...")

        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model_name": config.EMBEDDING_MODEL_NAME},
        )

        texts = [c["text"] for c in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        ids = [c["id"] for c in chunks]
        metadatas = [
            {"source": c["source"], "categorie": c.get("categorie", "")}
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"[VectorDB] {len(chunks)} chunks indexes.")

    def _reload_existing_collection(self):
        print(f"[VectorDB] Rechargement de la collection '{self.collection_name}'...")

        self.collection = self.client.get_collection(name=self.collection_name)

        stored_model_name = self.collection.metadata.get("embedding_model_name")
        if stored_model_name is None:
            raise RuntimeError(
                "La collection existe mais ne contient pas de metadonnee "
                "'embedding_model_name'."
            )

        self.embedding_model = SentenceTransformer(stored_model_name)
        print(f"[VectorDB] Modele d'embedding recharge : {stored_model_name}")

    def _encode(self, text):
        return self.embedding_model.encode(
            [text],
            normalize_embeddings=True,
        )[0].tolist()

    def retrieve(self, question, n=config.N_CHUNKS_RETRIEVED):
        query_embedding = self._encode(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class LemmaMemory:
    def __init__(self):
        self.index = faiss.IndexFlatL2(768)
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.lemmas = []
        
    def add_lemma(self, lemma, proof):
        emb = self.encoder.encode([lemma])[0]
        self.index.add(np.array([emb]))
        self.lemmas.append((lemma, proof))
        
    def retrieve(self, query, k=3):
        emb = self.encoder.encode([query])[0]
        distances, indices = self.index.search(np.array([emb]), k)
        return [self.lemmas[i] for i in indices[0]]
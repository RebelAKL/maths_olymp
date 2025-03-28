import torch
import numpy as np
from scipy.stats import entropy
from typing import List

class UncertaintyQuantifier:
    def __init__(self, model, num_samples=10):
        self.model = model
        self.num_samples = num_samples
        
    def monte_carlo_dropout(self, prompt: str) -> List[str]:
        """Generate multiple samples with dropout enabled"""
        original_mode = self.model.training
        self.model.train()  # Enable dropout
        
        samples = []
        for _ in range(self.num_samples):
            output = self.model.generate([prompt])[0].outputs[0].text
            samples.append(output)
            
        self.model.eval()  # Restore original mode
        return samples
    
    def calculate_confidence(self, samples: List[str]) -> float:
        """Compute semantic similarity confidence score"""
        # Get embeddings for all samples
        embeddings = self.model.get_embeddings(samples)
        
        # Compute pairwise cosine similarities
        similarity_matrix = np.zeros((len(samples), len(samples)))
        for i in range(len(samples)):
            for j in range(len(samples)):
                similarity_matrix[i,j] = cosine_similarity(
                    embeddings[i], embeddings[j])
                
        # Compute entropy-based uncertainty
        avg_similarity = np.mean(similarity_matrix)
        diversity = 1 - avg_similarity
        return max(0.0, 1.0 - diversity)
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
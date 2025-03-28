import logging
from typing import List, Tuple
from .lemma_bank import LemmaMemory

class MemoryManager:
    def __init__(self, config: TrainConfig):
        self.memory = LemmaMemory()
        self.config = config
        self.access_counter = {}
        
    def add_lemmas(self, lemmas: List[Tuple[str, str]]):
        """Add new lemmas to memory with usage tracking"""
        for lemma, proof in lemmas:
            if lemma not in self.access_counter:
                self.memory.add_lemma(lemma, proof)
                self.access_counter[lemma] = 1
            else:
                self.access_counter[lemma] += 1
                
        # Prune memory if needed
        if len(self.memory.lemmas) > self.config.max_memory_size:
            self._prune_memory()
            
    def _prune_memory(self):
        """Remove least-used lemmas based on access counter"""
        sorted_lemmas = sorted(self.access_counter.items(), 
                             key=lambda x: x[1])
        remove_count = len(self.memory.lemmas) - self.config.max_memory_size
        for lemma, _ in sorted_lemmas[:remove_count]:
            index = next(i for i, (l, _) in enumerate(self.memory.lemmas) 
                      if l == lemma)
            del self.memory.lemmas[index]
            del self.access_counter[lemma]
            self.memory.index.remove_ids(np.array([index]))
            
    def query_memory(self, problem: str) -> List[str]:
        """Retrieve relevant lemmas with usage tracking"""
        lemmas = self.memory.retrieve(problem)
        for lemma, _ in lemmas:
            self.access_counter[lemma] += 1
        return lemmas
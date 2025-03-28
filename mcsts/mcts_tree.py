class MCTSNode:
    def __init__(self, state, lemma_memory, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.lemma_memory = lemma_memory
        self.lemmas = self._retrieve_lemmas()
        
    def _retrieve_lemmas(self):
        return self.lemma_memory.retrieve(self.state)
    
    def expand(self, llm_engine):
        prompts = [f"Current state: {self.state}\nPossible next step:"]
        outputs = llm_engine.generate(prompts)
        actions = [output.outputs[0].text for output in outputs]
        self.children = [MCTSNode(f"{self.state}\n{action}", 
                        self.lemma_memory, self) 
                       for action in actions]
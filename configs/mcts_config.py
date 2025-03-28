class MCTSConfig:
    def __init__(self):
        self.num_simulations = 200
        self.exploration_constant = 1.41  # sqrt(2)
        self.max_depth = 15
        self.action_space_size = 25
        self.temperature = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        self.lemma_retrieval_topk = 3
        self.environment_timeout = 30  # seconds
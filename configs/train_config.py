class TrainConfig:
    def __init__(self):
        # Base training params
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 5
        self.warmup_steps = 100
        self.weight_decay = 0.01
        
        # RL specific
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_clip = 0.2
        self.entropy_coeff = 0.01
        
        # Curriculum learning
        self.curriculum_thresholds = [0.65, 0.75, 0.85]
        self.difficulty_levels = ['basic', 'intermediate', 'advanced']
        
        # Memory management
        self.max_memory_size = 10000
        self.memory_prune_interval = 1000
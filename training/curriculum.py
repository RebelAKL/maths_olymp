class CurriculumManager:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.current_level = 0
        self.performance_history = []
        
    def update_level(self, recent_performance: float):
        """Adjust difficulty based on recent success rate"""
        avg_performance = np.mean(self.performance_history[-10:] 
                          if self.performance_history else 0.0)
        
        if avg_performance > self.config.curriculum_thresholds[self.current_level]:
            if self.current_level < len(self.config.difficulty_levels) - 1:
                self.current_level += 1
        elif avg_performance < self.config.curriculum_thresholds[self.current_level] - 0.1:
            if self.current_level > 0:
                self.current_level -= 1
                
    def get_current_dataset(self):
        return self.config.difficulty_levels[self.current_level]
    
    def record_performance(self, success: bool):
        self.performance_history.append(1.0 if success else 0.0)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
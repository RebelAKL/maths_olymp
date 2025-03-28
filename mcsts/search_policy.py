import math
import numpy as np
from typing import List
from .mcts_tree import MCTSNode

class MCTSSearchPolicy:
    def __init__(self, config: MCTSConfig):
        self.config = config
        
    def select_action(self, node: MCTSNode) -> MCTSNode:
        """UCT selection"""
        total_visits = sum(child.visits for child in node.children)
        
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            exploit = child.value / (child.visits + 1e-9)
            explore = math.sqrt(math.log(total_visits + 1) / 
                              (child.visits + 1e-9))
            score = exploit + self.config.exploration_constant * explore
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand_node(self, node: MCTSNode, llm_engine, lemma_memory):
        # Get LLM generated actions
        prompt = f"Current state: {node.state}\nPossible next steps:"
        llm_actions = llm_engine.generate([prompt])[0].outputs[0].text
        llm_actions = llm_actions.split('\n')[:self.config.action_space_size]
        
        # Get lemma-based actions
        lemma_actions = [lemma[0] for lemma in lemma_memory.query(node.state)]
        
        # Combine and deduplicate
        all_actions = list(set(llm_actions + lemma_actions))
        return all_actions[:self.config.action_space_size]
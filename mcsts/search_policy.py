import math
from typing import List
from mcts_tree import GuidedMCTSNode
from configs import mcts_config


class GuidedMCTSSearchPolicy:
    def __init__(self, config):
        """
        Config should be a dict or object containing at least:
            - 'exploration_constant': float
            - 'action_space_size': int
        """
        self.config = config

    def select_action(self, node: GuidedMCTSNode) -> GuidedMCTSNode:
        """
        Use the Upper Confidence Bound for Trees (UCT) to select the best child node.
        """
        total_visits = sum(child.visits for child in node.children)
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            exploit = child.Q_value / (child.visits + 1e-9)
            explore = math.sqrt(math.log(total_visits + 1) / (child.visits + 1e-9))
            score = exploit + self.config['exploration_constant'] * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand_node(self, node: GuidedMCTSNode, llm_engine, lemma_memory) -> List[str]:
        """
        Generate candidate actions from both the LLM and lemma memory,
        combine and deduplicate them.
        """
        # LLM-generated actions
        prompt = f"Current state:\n{node.query}\n\nList possible next steps:"
        llm_output = llm_engine.generate([prompt])
        llm_actions = [action.strip() for action in llm_output[0].outputs[0].text.strip().split('\n')]
        llm_actions = llm_actions[:self.config['action_space_size']]

        # Lemma-based actions (assuming lemma_memory.query returns a list of tuples)
        lemma_actions = [lemma[0] for lemma in lemma_memory.query(node.query)]
        all_actions = list(set(llm_actions + lemma_actions))
        return all_actions[:self.config['action_space_size']]
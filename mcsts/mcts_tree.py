# guided_mcts_node.py
from collections import deque, defaultdict
from difflib import SequenceMatcher
import re
from typing import List, Set, Optional, Union

# Import Pydantic for data validation of LLM responses
from pydantic import BaseModel, Field

# ----------------------------
# Pydantic models for guided decoding with vLLM
# ----------------------------
class ScoreModel(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Score between 0 and 100")

class FeedbackModel(BaseModel):
    issues: Union[str, List[str]]

# ----------------------------
# Simplified LLM Interface
# ----------------------------
class LLMInterface:
    INSTRUCTION_SET = 'llama3'
    
    @classmethod
    def ask_llm_remote(cls, prompt: str, model_schema=None):
        """
        Stub implementation for asking the LLM a question.
        In production, this should call your LLM API and return its response.
        """
        # For demonstration, simply return a dummy score.
        return "75"
    
    @classmethod
    def generate(cls, prompts: List[str]):
        """
        Stub implementation for generating LLM responses.
        Returns a list of dummy responses, each with an 'outputs' attribute.
        """
        class DummyOutput:
            def __init__(self, text: str):
                self.text = text
                
        class DummyResponse:
            def __init__(self, text: str):
                self.outputs = [DummyOutput(text)]
                
        # For simplicity, each prompt returns a newline-separated list of steps.
        dummy_text = "Step 1\nStep 2\nStep 3"
        return [DummyResponse(dummy_text) for _ in prompts]
    
    @classmethod
    def set_instruction_set(cls, instruction_set: str):
        if instruction_set in ["alpaca", "vicuna", "llama3", "chatml"]:
            cls.INSTRUCTION_SET = instruction_set
        else:
            raise ValueError(f"Unsupported instruction set: {instruction_set}")

# ----------------------------
# Base Node class (from the notebook)
# ----------------------------
class Node:
    node_counter = 0

    def __init__(self, query: str, answer: str, feedback: Optional[dict] = None,
                 refined_answer: Optional[str] = None, parent: Optional["Node"] = None):
        self.id = self.generate_id(parent)
        self.query = query
        self.answer = answer
        self.feedback = feedback
        self.refined_answer = refined_answer
        self.parent = parent
        self.children: Set["Node"] = set()
        self.visits = 0
        self.rewards = deque(maxlen=100)  # Store only the last 100 rewards
        self.reward_sum = 0
        self.reward_sum_squared = 0
        self.Q_value = 0
        self.previous_Q_value = 0
        self.max_children = 5
        self.importance_weight = 1.0
        self.depth = 0 if parent is None else parent.depth + 1

    @classmethod
    def generate_id(cls, parent: Optional["Node"]):
        if parent is None:
            cls.node_counter += 1
            return str(cls.node_counter - 1)
        else:
            parent_id = parent.id
            child_number = len(parent.children) + 1
            return f"{parent_id}.{child_number}"

    def __str__(self):
        return f"Node_{self.id}"

    def increment_visits(self):
        self.visits += 1

    def update_Q(self):
        self.previous_Q_value = self.Q_value
        if self.rewards:
            min_reward = min(self.rewards)
            avg_reward = sum(self.rewards) / len(self.rewards)
            self.Q_value = 0.5 * (min_reward + avg_reward)
        else:
            if not self.children:
                self.Q_value = 0
            else:
                self.Q_value = max(child.Q_value for child in self.children)
        self.Q_value = max(0, min(100, self.Q_value))

    def add_reward(self, reward: float):
        if len(self.rewards) == self.rewards.maxlen:
            old_reward = self.rewards[0]
            self.reward_sum -= old_reward
            self.reward_sum_squared -= old_reward ** 2
        self.rewards.append(reward)
        self.reward_sum += reward
        self.reward_sum_squared += reward ** 2

    def get_ancestors(self) -> List["Node"]:
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def analyze_historical_performance(self) -> Optional[str]:
        ancestors = self.get_ancestors()[:5]  # Limit to 5 most recent ancestors
        issues = defaultdict(float)

        def normalize_text(text: str) -> str:
            return re.sub(r'[^\w\s]', '', text.lower())

        def fuzzy_match(a: str, b: str, threshold=0.8) -> bool:
            return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio() > threshold

        for i, ancestor in enumerate(ancestors):
            if ancestor.feedback and 'issues' in ancestor.feedback:
                weight = 1 / (i + 1)
                for issue in ancestor.feedback['issues']:
                    issue_text = issue if isinstance(issue, str) else str(issue)
                    issue_text = normalize_text(issue_text)
                    matched = False
                    for existing_issue in issues:
                        if fuzzy_match(issue_text, existing_issue):
                            issues[existing_issue] += weight
                            matched = True
                            break
                    if not matched:
                        issues[issue_text] = weight

        if issues:
            common_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:3]
            formatted_issues = "\n".join([f"- {issue}" for issue, _ in common_issues])
            return f"[Common Issues]\n{formatted_issues}"
        return None

    def generate_feedback(self):
        # Stub: In production, implement feedback generation via LLM prompts.
        self.feedback = {"issues": ["example issue"]}

    def refine_answer(self):
        historical_insights = self.analyze_historical_performance()
        historical_context = ""
        if historical_insights:
            historical_context = f"Note: Previous answers often struggled with:\n{historical_insights}."
        base_answer = self.parent.refined_answer if self.parent and self.parent.refined_answer else (
            self.parent.answer if self.parent else self.answer)
        # Stub: In production, refine the answer using an LLM prompt.
        self.refined_answer = f"Refined: {base_answer} with context: {historical_context}"

    def self_evaluate(self, scoring_method=None) -> float:
        # Stub: In production, evaluate self using an LLM or other evaluator.
        return 50.0

    def create_child(self) -> "Node":
        new_node = Node(
            query=self.query,
            answer=self.refined_answer if self.refined_answer else self.answer,
            feedback=None,
            refined_answer=None,
            parent=self
        )
        self.children.add(new_node)
        return new_node

# ----------------------------
# GuidedMCTSNode: Extends Node with guided decoding and MCTS functionalities.
# ----------------------------
class GuidedMCTSNode(Node):
    def __init__(self, query: str, answer: str, lemma_memory, parent: Optional[Node] = None):
        super().__init__(query, answer, parent=parent)
        self.lemma_memory = lemma_memory
        self.lemmas = self.retrieve_lemmas()

    def retrieve_lemmas(self) -> List[str]:
        # Use your lemma memory interface to fetch relevant lemmas based on the query.
        return self.lemma_memory.retrieve(self.query)

    def expand(self, llm_engine: LLMInterface, action_space_size: int) -> List["GuidedMCTSNode"]:
        """
        Expand the current node by generating candidate next steps using the LLM.
        """
        prompt = f"Current state:\n{self.query}\n\nList {action_space_size} possible next steps (one per line):"
        llm_output = llm_engine.generate([prompt])
        actions = [action.strip() for action in llm_output[0].outputs[0].text.strip().split('\n')]
        actions = actions[:action_space_size]

        new_children = []
        for action in actions:
            new_query = f"{self.query}\n{action}"
            child_node = GuidedMCTSNode(new_query, self.answer, self.lemma_memory, parent=self)
            self.children.add(child_node)
            new_children.append(child_node)
        return new_children

    def rollout(self, llm_engine: LLMInterface, max_depth: int = 5) -> float:
        """
        Perform a rollout (simulation) from the current node using the LLM until a terminal condition is met.
        Returns a score for the simulated rollout.
        """
        current_query = self.query
        for _ in range(max_depth):
            prompt = f"Current state:\n{current_query}\n\nWhat is the next step to solve this problem? If no further steps, say 'STOP'."
            llm_output = llm_engine.generate([prompt])
            action = llm_output[0].outputs[0].text.strip()
            if action.upper() == 'STOP' or not action:
                break
            current_query = f"{current_query}\n{action}"
        
        evaluation_prompt = f"Evaluate the following solution state for correctness and completeness:\n{current_query}\n\nProvide a score between 0 and 100:"
        score_response = llm_engine.ask_llm_remote(evaluation_prompt, model_schema=ScoreModel)
        try:
            score = float(score_response)
        except Exception:
            score = 0.0
        return score

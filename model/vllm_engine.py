from vllm import LLM, SamplingParams
from configs.model_config import MODEL_CONFIG

class MathLLMEngine:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_CONFIG["model_name"],
            tensor_parallel_size=MODEL_CONFIG["tensor_parallel_size"],
            dtype=MODEL_CONFIG["dtype"]
        )
        self.sampling_params = MODEL_CONFIG["sampling_params"]
    
    def generate(self, prompts):
        return self.llm.generate(prompts, self.sampling_params)
    
    def get_embeddings(self, texts):
        return self.llm.encode(texts)
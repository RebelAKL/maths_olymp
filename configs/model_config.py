from vllm import SamplingParams

MODEL_CONFIG = {
    "model_name": "deepseek-ai/deepseek-math-7b-base",
    "tensor_parallel_size": 2,
    "dtype": "bfloat16",
    "sampling_params": SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
}
import requests
import json
from requests.exceptions import RequestException
from typing import Optional, Type
from pydantic import BaseModel

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class APIManager(metaclass=SingletonMeta):
    def __init__(self):
        self.session = requests.Session()
        
    def _api_request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            raise Exception(f'API request error: {e}')
            
    def run(self, prompt: str, temperature: float = 0.0, api_url: str = "http://localhost:8000",
            model_schema: Optional[Type[BaseModel]] = None): 
        """
        Send completion request to vLLM API server
        
        Parameters:
        - prompt: text prompt to generate from
        - temperature: controls randomness (0.0 = deterministic)
        - api_url: URL of the vLLM API server (on Kaggle use ngrok to get the reverse proxy URL)
        - output_model: optional pydantic model to validate the JSON output (e.g., ScoreModel)
        """
        request_payload = {
            'prompt': prompt,
            'max_tokens': 2048,
            'temperature': temperature,
            'top_p': 0.9,
            'min_p': 0.0,
            'top_k': 0,
            'typical_p': 1.0,
            'tfs': 1.0,
            'top_a': 0.0,
            'repetition_penalty': 1.0,
            'min_new_tokens': 200,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 8192,
            'ban_eos_token': False,
            'skip_special_tokens': True,
        }

        # Add output_schema only if model_schema is provided
        if model_schema:
            request_payload['output_schema'] = model_schema.model_json_schema()
        
        response = self._api_request('POST', f'{api_url}/v1/completions', json=request_payload)
        generated_text = response['choices'][0]['text']

        if model_schema:
            try:
                # When using model_schema, the response is guaranteed to be valid JSON
                return json.loads(generated_text)
            except json.JSONDecodeError as e:
                # This should only happen if there's an unexpected issue with the API
                raise ValueError(f"Failed to parse response as JSON: {e}. Response: {generated_text[:100]}...")
        else:
            return generated_text

api_manager = APIManager()
import fnmatch
from pathlib import Path

MODEL_TYPE_RULES = {
    "gpt-4o-2024-11-20": "close",
    "claude-3-opus": "close",
    
    "gpt-4*": "close",  
    "claude-*": "close",    
    
    "vicuna-*": "open",   
    "Qwen-*-Chat": "open",  
    "Meta-Llama-3-*-Instruct": "open",
    "openchat-*": "open",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together",
    "Qwen2.5-*B-Instruct": "open",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "mistralai/Mistral-7B-Instruct-v0.1": "together",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "together",
    "google/gemma-2-9b-it": "together",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "deepseek-ai/DeepSeek-V3": "together",
    "Skywork-Critic-Llama-3.1-8B": "open",
    "Llama-3.1-8B-Instruct": "open",
    "JudgeLM-13B-v1.0": "open",
    "JudgeLM-7B-v1.0": "open",
    "prometheus-7b-v2.0":"open",
    "Mistral-7B-Instruct-v0.2":"open",
    "Meta-Llama-3.1-8B-Instruct":"open",
    "PandaLM-7B-v1": "open",
    "autoj-13b": "open",
    "deepseek-ai/DeepSeek-R1":"together"

}

def infer_model_type(model_id: str) -> str:

    if model_id in MODEL_TYPE_RULES:
        return MODEL_TYPE_RULES[model_id]
    
    for pattern, model_type in MODEL_TYPE_RULES.items():
        if '*' in pattern and fnmatch.fnmatch(model_id, pattern):
            return model_type
    
    if Path(model_id).exists():
        return "open"
    
    if '/' in model_id and len(model_id.split('/')) == 2:
        return "open"
    
    raise ValueError(f"Cannot recognize the model type: {model_id}")
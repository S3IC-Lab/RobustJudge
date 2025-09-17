from abc import ABC, abstractmethod
from model.model_adapter import load_model_demo
from model.inference import (
    api_generate, api_generate_loss,
    open_model_generate, open_model_generate_for_multi_responses,
    open_model_generate_with_logprobs
)
from model.model_type_infer import infer_model_type
from transformers import AutoModelForCausalLM, AutoTokenizer
import tiktoken
from transformers import AutoTokenizer
import tiktoken

def get_compatible_tokenizer(model_name: str):
    """
    function:
    1. try tiktoken first
    2. if 1 fail then try HF
    3. if 2 fail, return None
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False
            )
        except Exception as e:
            return None

class BaseModel(ABC):
    @abstractmethod
    def model_generate(self, prompt):
        """
        Generate a response from the model given a prompt.
        """

class OpenSourceModel(BaseModel):
    def __init__(self, model,model_id, tokenizer,args):
        """
        Initialize the open-source model with the given model and tokenizer.
        """
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
    
    def model_generate(self, prompt):
        return open_model_generate(self.model,self.tokenizer,prompt,self.model_id,self.args)
    
    def model_generate_for_multi_responses(self, prompt, num_return_sequences):
        return open_model_generate_for_multi_responses(self.model,self.tokenizer,prompt,self.model_id,self.args, num_return_sequences)
    
    def model_generate_loss(self, prompt, max_tokens=1,
            temperature=0.0, top_p=1.0, logprobs=True, top_logprobs=20, seed=0):
        return open_model_generate_with_logprobs(
            model = self.model,
            model_id = self.model_id,
            tokenizer = self.tokenizer,
            prompt = prompt,
            max_tokens = max_tokens,
            temperature= temperature,
            top_p = top_p,
            logprobs = logprobs,
            top_logprobs = top_logprobs,
            do_sample = False,
            seed = seed
        )
    
class ClosedSourceModel(BaseModel):
    def __init__(self, model_id: str, temperature: str, max_token: int):
        """
        Initialize the closed-source model with an API key.
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_token = max_token
        self.tokenizer = get_compatible_tokenizer(model_id)

        self.base_url=""
        self.api_key=""
        
    def model_generate(self, prompt):
        return api_generate(
            prompt,
            self.model_id,
            self.temperature,
            self.max_token,
            self.base_url,
            self.api_key,
            num_return_sequences=1
        )
    def model_generate_for_multi_responses(self, prompt, num_return_sequences):
        return api_generate(
            prompt,
            self.model_id,
            self.temperature,
            self.max_token,
            self.base_url,
            self.api_key,
            num_return_sequences
        )
    
    def model_generate_loss(self, prompt, max_tokens=1,
            temperature=0.0, top_p=1.0, logprobs=True, top_logprobs=20, seed=0):
        return api_generate_loss(
            prompt = prompt,
            model_id = self.model_id,
            base_url = self.base_url,
            api_key = self.api_key, 
            max_tokens = max_tokens, 
            temperature = temperature,
            top_p=top_p, 
            logprobs=logprobs, 
            top_logprobs=top_logprobs, 
            seed=seed
        )

class TogetherAiModel(BaseModel):
    def __init__(self, model_id: str, temperature: str, max_token: int):
        """
        Initialize the TogetherAI model with an API key.
        """
        self.model_id = model_id
        # self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model = None
        self.tokenizer = get_compatible_tokenizer(model_id)
        self.temperature = temperature
        self.max_token = max_token
        self.base_url = ""
        self.api_key = ""
    def model_generate(self, prompt):
        return api_generate(
            prompt,
            self.model_id,
            self.temperature,
            self.max_token,
            self.base_url,
            self.api_key,
            num_return_sequences=1
        )
    def model_generate_for_multi_responses(self, prompt, num_return_sequences):
        return api_generate(
            prompt,
            self.model_id,
            self.temperature,
            self.max_token,
            self.base_url,
            self.api_key,
            num_return_sequences
        )
    def model_generate_loss(self, prompt, max_tokens=1,
            temperature=0.0, top_p=1.0, logprobs=True, top_logprobs=20, seed=0):
        return api_generate_loss(
            prompt = prompt,
            model_id = self.model_id,
            base_url = self.base_url,
            api_key = self.api_key, 
            max_tokens = max_tokens, 
            temperature = temperature,
            top_p=top_p, 
            logprobs=logprobs, 
            top_logprobs=top_logprobs, 
            seed=seed
        )



def load_model_instance(args, model_path:str,model_id: str,temperature: str,
                        max_token: int):
    """
    Load a model (target or attack model) based on the configuration (open or closed source).

    Args:
        args: Configuration arguments.
        model_type: Type of the model ('target' or 'attack').
        model_id: ID or model identifier (for closed-source models).

    Returns:
        An instance of the appropriate model (either open-source or closed-source).
    """
    model_type = infer_model_type(model_id)
    print(f"model_type is {model_type}")
    if model_type == "close":
        # Load a closed-source model with an API key or credentials
        return ClosedSourceModel(model_id=model_id,temperature=temperature,max_token=max_token)
    elif model_type == "open":
        # Load an open-source model with model and tokenizer
        model, tokenizer = load_model_demo(args, model_path)
        return OpenSourceModel(model, model_id,tokenizer,args)

    else:
        print("TOGETHER MODEL")
        return TogetherAiModel(model_id=model_id,temperature=temperature,max_token=max_token)

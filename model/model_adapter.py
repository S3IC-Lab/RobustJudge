"""
Model adapter registration.
Adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py.
"""

import math
import os
import re
import sys
from typing import Dict, List, Optional
import warnings

from model.model_compression import load_compress_model
from model.template.conversation import Conversation, get_conv_template
from model.model_chatglm import generate_stream_chatglm
from model.model_inference import generate_stream
from model.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from model.common_utils import get_gpu_memory
import torch

import psutil
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

# CPU Instruction Set Architecture
CPU_ISA = os.getenv("CPU_ISA")

# Check an environment variable to check if we should be sharing Peft model
# weights.  When false we treat all Peft models as separate.
peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)

ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
)

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())

@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")

def raise_warning_for_incompatible_cpu_offloading_configuration(
    device: str, load_8bit: bool, cpu_offloading: bool):
    if cpu_offloading == False:
        return
    # if cpu_offloading:
    if not load_8bit:
        warnings.warn(
            "The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
            "Use '--load-8bit' to enable 8-bit-quantization\n"
            "Continuing without cpu-offloading enabled\n"
        )
        return False
    if not "linux" in sys.platform:
        warnings.warn(
            "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
            "Continuing without cpu-offloading enabled\n"
        )
        return False
    if device != "cuda":
        warnings.warn(
            "CPU-offloading is only enabled when using CUDA-devices\n"
            "Continuing without cpu-offloading enabled\n"
        )
        return False
    return cpu_offloading

def load_model(
    model_path: str,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    # gptq_config: Optional[GptqConfig] = None,
    # awq_config: Optional[AWQConfig] = None,
    # exllama_config: Optional[ExllamaConfig] = None,
    # xft_config: Optional[XftConfig] = None,
    revision: str = "main",
    debug: bool = False,
):
    """Load a model from Hugging Face."""
    import accelerate

    # get model adapter
    adapter = get_model_adapter(model_path)

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
        if CPU_ISA in ["avx512_bf16", "amx"]:
            try:
                import intel_extension_for_pytorch as ipex

                kwargs = {"torch_dtype": torch.bfloat16}
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                )
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        import transformers

        version = tuple(int(v) for v in transformers.__version__.split("."))
        if version < (4, 35, 0):
            # NOTE: Recent transformers library seems to fix the mps issue, also
            # it has made some changes causing compatibility issues with our
            # original patch. So we only apply the patch for older versions.

            # Avoid bugs in mps backend by not using in-place operations.
            replace_llama_attn_with_non_inplace_operations()
    elif device == "xpu":
        kwargs = {"torch_dtype": torch.bfloat16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            warnings.warn(
                "Intel Extension for PyTorch is not installed, but is required for xpu inference."
            )
    elif device == "npu":
        kwargs = {"torch_dtype": torch.float16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import torch_npu
        except ImportError:
            warnings.warn("Ascend Extension for PyTorch is not installed.")
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )
            if debug:
                print(model)
            return model, tokenizer
    
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype

    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)

    if (
        device == "cpu"
        and kwargs["torch_dtype"] is torch.bfloat16
        and CPU_ISA is not None
    ):
        model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device in (
        "mps",
        "xpu",
        "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    if debug:
        print(model)

    return model, tokenizer

def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

def get_generate_stream_function(model: torch.nn.Module, model_path: str):
    """Get the generate_stream function for inference."""

    model_type = str(type(model)).lower()
    is_chatglm = "chatglm" in model_type

    if is_chatglm:
        return generate_stream_chatglm
    else:
        return generate_stream

def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]

class VicunaAdapter(BaseModelAdapter):
    "Model adapter for Vicuna models (e.g., lmsys/vicuna-7b-v1.5)" ""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        self.raise_warning_for_old_weights(model)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.3: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommended).\n"
            )

class QwenChatAdapter(BaseModelAdapter):
    """The model adapter for Qwen/Qwen-7B-Chat
    To run this model, you need to ensure additional flash attention installation:
    ``` bash
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention && pip install .
    pip install csrc/layer_norm
    pip install csrc/rotary
    ```

    Since from 2.0, the following change happened
    - `flash_attn_unpadded_func` -> `flash_attn_varlen_func`
    - `flash_attn_unpadded_qkvpacked_func` -> `flash_attn_varlen_qkvpacked_func`
    - `flash_attn_unpadded_kvpacked_func` -> `flash_attn_varlen_kvpacked_func`
    You may need to revise the code in: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py#L69
    to from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    """

    def match(self, model_path: str):
        return "qwen" in model_path.lower()

    def float_set(self, config, option):
        config.bf16 = False
        config.fp16 = False
        config.fp32 = False

        if option == "bf16":
            config.bf16 = True
        elif option == "fp16":
            config.fp16 = True
        elif option == "fp32":
            config.fp32 = True
        else:
            print("Invalid option. Please choose one from 'bf16', 'fp16' and 'fp32'.")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from transformers.generation import GenerationConfig

        revision = from_pretrained_kwargs.get("revision", "main")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # NOTE: if you use the old version of model file, please remove the comments below
        # config.use_flash_attn = False
        self.float_set(config, "fp16")
        generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        ).eval()
        if hasattr(model.config, "use_dynamic_ntk") and model.config.use_dynamic_ntk:
            model.config.max_sequence_length = 16384
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision
        )
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.pad_token_id = generation_config.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("qwen-7b-chat")

class Llama2Adapter(BaseModelAdapter):
    """The model adapter for Llama-2 (e.g., meta-llama/Llama-2-7b-hf)"""

    def match(self, model_path: str):
        return "llama-2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

class Llama3Adapter(BaseModelAdapter):
    """The model adapter for Llama-3 (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"""

    def match(self, model_path: str):
        return "llama-3-" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-3")

class Llama31Adapter(BaseModelAdapter):
    """The model adapter for Llama-3 (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"""

    def match(self, model_path: str):
        keywords = [
            "llama-3.1",
        ]
        for keyword in keywords:
            if keyword in model_path.lower():
                return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if model_path.lower() in [
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
            "the-real-chatbot-v2",
        ]:
            return get_conv_template("meta-llama-3.1-sp")
        return get_conv_template("meta-llama-3.1")

class DeepseekCoderAdapter(BaseModelAdapter):
    """The model adapter for deepseek-ai's coder models"""

    def match(self, model_path: str):
        return "deepseek-coder" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-coder")

class DeepseekChatAdapter(BaseModelAdapter):
    """The model adapter for deepseek-ai's chat models"""

    # Note: that this model will require tokenizer version >= 0.13.3 because the tokenizer class is LlamaTokenizerFast

    def match(self, model_path: str):
        return "deepseek-llm" in model_path.lower() and "chat" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("deepseek-chat")

class ChatGLMAdapter(BaseModelAdapter):
    """The model adapter for THUDM/chatglm-6b, THUDM/chatglm2-6b"""

    def match(self, model_path: str):
        return "chatglm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        if "chatglm3" in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                encode_special_tokens=True,
                trust_remote_code=True,
                revision=revision,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, revision=revision
            )
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        if "chatglm3" in model_path.lower():
            return get_conv_template("chatglm3")
        return get_conv_template("chatglm")

class OpenChat35Adapter(BaseModelAdapter):
    """The model adapter for OpenChat 3.5 (e.g. openchat/openchat_3.5)"""

    def match(self, model_path: str):
        if "openchat" in model_path.lower() and "3.5" in model_path.lower():
            return True
        elif "starling-lm" in model_path.lower():
            return True
        return False

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openchat_3.5")

class ChatGPTAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return model_path in OPENAI_MODEL_LIST

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "browsing" in model_path or "o1" in model_path:
            return get_conv_template("api_based_default")
        if "gpt-4-turbo-2024-04-09" in model_path or "gpt-4o-2024-05-13" in model_path or "gpt-4o-2024-08-06" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt2-chatbot" in model_path or "anonymous-chatbot" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "chatgpt-4o-latest" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt-mini" in model_path or "gpt-4o-mini-2024-07-18" in model_path:
            return get_conv_template("gpt-mini")
        return get_conv_template("chatgpt")

class MistralAdapter(BaseModelAdapter):
    """The model adapter for Mistral AI models"""

    def match(self, model_path: str):
        return "mistral" in model_path.lower() or "mixtral" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mistral")

      
# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(VicunaAdapter)
register_model_adapter(QwenChatAdapter)
register_model_adapter(Llama2Adapter)
register_model_adapter(Llama3Adapter)
register_model_adapter(Llama31Adapter)
register_model_adapter(DeepseekCoderAdapter)
register_model_adapter(DeepseekChatAdapter)
register_model_adapter(ChatGLMAdapter)
register_model_adapter(OpenChat35Adapter)
register_model_adapter(ChatGPTAdapter)
register_model_adapter(MistralAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)



def load_model_demo(args,model_path):
    return load_model(
        model_path,
        revision=args.revision,
        device="cuda",
        num_gpus=args.num_gpus_per_model,
        max_gpu_memory=args.max_gpu_memory,
        dtype=torch.float16,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
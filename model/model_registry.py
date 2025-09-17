"""
Additional information of the models.
Adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_registry.py.
"""
from collections import namedtuple, OrderedDict
from typing import List


ModelInfo = namedtuple("ModelInfo", ["simple_name", "link", "description"])


model_info = OrderedDict()


def register_model_info(
    full_names: List[str], simple_name: str, link: str, description: str
):
    info = ModelInfo(simple_name, link, description)

    for full_name in full_names:
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    if name in model_info:
        return model_info[name]
    else:
        # To fix this, please use `register_model_info` to register your model
        return ModelInfo(
            name, "", "Register the description at fastchat/model/model_registry.py"
        )

register_model_info(
    [
        "chatgpt-4o-latest",
        "chatgpt-4o-latest-20240903",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
    ],
    "GPT-4o",
    "https://openai.com/index/hello-gpt-4o/",
    "The flagship model across audio, vision, and text by OpenAI",
)

register_model_info(
    [
        "claude-3-5-sonnet-20240620",
    ],
    "Claude 3.5",
    "https://www.anthropic.com/news/claude-3-5-sonnet",
    "Claude by Anthropic",
)

register_model_info(
    [
        "llama-3.2-vision-90b-instruct",
        "llama-3.2-vision-11b-instruct",
        "llama-3.2-3b-instruct",
        "llama-3.2-1b-instruct",
        "llama-3.1-405b-instruct-bf16",
        "llama-3.1-405b-instruct-fp8",
        "llama-3.1-405b-instruct",
        "llama-3.1-70b-instruct",
        "llama-3.1-8b-instruct",
    ],
    "Llama 3.1",
    "https://llama.meta.com/",
    "Open foundation and chat models by Meta",
)

register_model_info(
    [
        "mistral-large-2407",
    ],
    "Mistral",
    "https://mistral.ai/news/mistral-large-2407/",
    "Mistral Large 2",
)

register_model_info(
    [
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt2-chatbot",
        "im-also-a-good-gpt2-chatbot",
        "im-a-good-gpt2-chatbot",
    ],
    "GPT-4-Turbo",
    "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
    "GPT-4-Turbo by OpenAI",
)

register_model_info(
    [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-2.1",
        "claude-2.0",
        "claude-1",
    ],
    "Claude",
    "https://www.anthropic.com/news/claude-3-family",
    "Claude by Anthropic",
)

register_model_info(
    ["deepseek-coder-v2", "deepseek-v2-api-0628", "deepseek-v2.5"],
    "DeepSeek Coder v2",
    "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "An advanced code model by DeepSeek",
)

register_model_info(
    ["llama-3-70b-instruct", "llama-3-8b-instruct"],
    "Llama 3",
    "https://ai.meta.com/blog/meta-llama-3/",
    "Open foundation and chat models by Meta",
)

register_model_info(
    [
        "qwen2.5-72b-instruct",
        "qwen2-72b-instruct",
        "qwen-max-0403",
        "qwen-max-0428",
        "qwen-max-0919",
        "qwen-plus-0828",
        "qwen2-vl-7b-instruct",
        "qwen-vl-max-0809",
    ],
    "Qwen Max",
    "https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction",
    "The Frontier Qwen Model by Alibaba",
)

register_model_info(
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0314",
        "gpt-3.5-turbo-0613",
    ],
    "GPT-3.5",
    "https://platform.openai.com/docs/models/gpt-3-5",
    "GPT-3.5-Turbo by OpenAI",
)

register_model_info(
    [
        "codestral-2405",
        "mixtral-8x22b-instruct-v0.1",
    ],
    "Mixtral of experts",
    "https://mistral.ai/news/mixtral-8x22b/",
    "A Mixture-of-Experts model by Mistral AI",
)

register_model_info(
    [
        "mixtral-8x7b-instruct-v0.1",
        "mistral-large-2402",
        "mistral-large-2407",
        "mistral-medium",
        "mistral-next",
        "mistral-7b-instruct-v0.2",
        "mistral-7b-instruct",
        "pixtral-12b-2409",
    ],
    "Mixtral of experts",
    "https://mistral.ai/news/mixtral-of-experts/",
    "A Mixture-of-Experts model by Mistral AI",
)

register_model_info(
    [
        "qwen1.5-110b-chat",
    ],
    "Qwen 1.5",
    "https://qwenlm.github.io/blog/qwen1.5-110b/",
    "The First 100B+ Model of the Qwen1.5 Series",
)

register_model_info(
    [
        "qwen1.5-72b-chat",
        "qwen1.5-32b-chat",
        "qwen1.5-14b-chat",
        "qwen1.5-7b-chat",
        "qwen1.5-4b-chat",
        "qwen1.5-1.8b-chat",
        "qwen1.5-0.5b-chat",
        "qwen-14b-chat",
    ],
    "Qwen 1.5",
    "https://qwenlm.github.io/blog/qwen1.5/",
    "A large language model by Alibaba Cloud",
)

register_model_info(
    ["glm-4-plus", "glm-4-0520", "glm-4-0116"],
    "GLM-4",
    "https://bigmodel.cn/dev/howuse/model",
    "Next-Gen Foundation Model by Zhipu AI",
)

register_model_info(
    ["qwen-14b-chat"],
    "Qwen",
    "https://huggingface.co/Qwen",
    "A large language model by Alibaba Cloud",
)

register_model_info(
    ["gpt-4-turbo-browsing"],
    "GPT-4-Turbo with browsing",
    "https://platform.openai.com/docs/assistants/overview",
    "GPT-4-Turbo with browsing by OpenAI",
)

register_model_info(
    ["gpt-4", "gpt-4-0314", "gpt-4-0613"],
    "GPT-4",
    "https://openai.com/research/gpt-4",
    "GPT-4 by OpenAI",
)

register_model_info(
    ["claude-instant-1", "claude-instant-1.2"],
    "Claude Instant",
    "https://www.anthropic.com/index/introducing-claude",
    "Claude Instant by Anthropic",
)

register_model_info(
    ["llama-2-70b-chat", "llama-2-34b-chat", "llama-2-13b-chat", "llama-2-7b-chat"],
    "Llama 2",
    "https://ai.meta.com/llama/",
    "Open foundation and fine-tuned chat models by Meta",
)

register_model_info(
    [
        "vicuna-33b",
        "vicuna-33b-v1.3",
        "vicuna-13b",
        "vicuna-13b-v1.5",
        "vicuna-7b",
        "vicuna-7b-v1.5",
    ],
    "Vicuna",
    "https://lmsys.org/blog/2023-03-30-vicuna/",
    "A chat assistant fine-tuned on user-shared conversations by LMSYS",
)

register_model_info(
    [
        "codellama-70b-instruct",
        "codellama-34b-instruct",
        "codellama-13b-instruct",
        "codellama-7b-instruct",
    ],
    "Code Llama",
    "https://ai.meta.com/blog/code-llama-large-language-model-coding/",
    "Open foundation models for code by Meta",
)

register_model_info(
    ["openchat-3.5-0106", "openchat-3.5"],
    "OpenChat 3.5",
    "https://github.com/imoneoi/openchat",
    "An open model fine-tuned on Mistral-7B using C-RLFT",
)

register_model_info(
    ["deepseek-llm-67b-chat"],
    "DeepSeek LLM",
    "https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat",
    "An advanced language model by DeepSeek",
)

register_model_info(
    ["llama2-70b-steerlm-chat"],
    "Llama2-70B-SteerLM-Chat",
    "https://huggingface.co/nvidia/Llama2-70B-SteerLM-Chat",
    "A Llama fine-tuned with SteerLM method by NVIDIA",
)

register_model_info(
    ["chatglm3-6b", "chatglm2-6b", "chatglm-6b"],
    "ChatGLM",
    "https://chatglm.cn/blog",
    "An open bilingual dialogue language model by Tsinghua University",
)

register_model_info(
    ["TinyLlama"],
    "TinyLlama",
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.",
)

register_model_info(
    ["llama-7b", "llama-13b"],
    "LLaMA",
    "https://arxiv.org/abs/2302.13971",
    "Open and efficient foundation language models by Meta",
)

register_model_info(
    ["open-llama-7b-v2-open-instruct", "open-llama-7b-open-instruct"],
    "Open LLaMa (Open Instruct)",
    "https://medium.com/vmware-data-ml-blog/starter-llm-for-the-enterprise-instruction-tuning-openllama-7b-d05fc3bbaccc",
    "Open LLaMa fine-tuned on instruction-following data by VMware",
)

register_model_info(
    ["Qwen-7B-Chat"],
    "Qwen",
    "https://huggingface.co/Qwen/Qwen-7B-Chat",
    "A multi-language large-scale language model (LLM), developed by Damo Academy.",
)

register_model_info(
    ["Llama2-Chinese-13b-Chat", "LLama2-Chinese-13B"],
    "Llama2-Chinese",
    "https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat",
    "A multi-language large-scale language model (LLM), developed by FlagAlpha.",
)

register_model_info(
    ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct"],
    "llama-3",
    "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    "Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes.",
)

register_model_info(
    ["stable-vicuna-13B-HF"],
    "stable-vicuna",
    "https://huggingface.co/TheBloke/stable-vicuna-13B-HF",
    "A Vicuna model fine-tuned using RLHF via PPO on various conversational and instructional datasets.",
)

register_model_info(
    ["Mistral-7B-OpenOrca"],
    "Open-Orca",
    "https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca",
    "A fine-tune of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) using [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)",
)



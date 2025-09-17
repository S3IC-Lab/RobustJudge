## How to use
1. To use `ClosedSourceModel` or `TogetherAiModel`, you need to add your `base_url`and `api_key` in the `RobustJudge/model/model_class.py`,as:
```
self.base_url="xxx"
self.api_key="xxx"
```
2. To use `OpenSourceModel`, you need to download the model in a local path.
## Model list
You can download the model and use the command below to test.
```
python test_chat.py --model-path "local_model's_path" --model-id "local_model's_name"
```
Or you can use the commands below to start chatting. It will automatically download the weights from Hugging Face repos. Downloaded weights are stored in a `.cache` folder in the user's home folder (e.g., `~/.cache/huggingface/hub/<model_name>`).
**NOTE**: you can use this command 
```
!export HF_ENDPOINT=https://hf-mirror.com
```
 if you cannot connect the huggingface in your area.
### Vicuna
[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) is based on Llama 2 and should be used under Llama's [model license](https://github.com/facebookresearch/llama/blob/main/LICENSE).

**NOTE: `transformers>=4.31` is required for 16K versions.**

| Size | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| 7B   | `python test_chat.py --model-path lmsys/vicuna-7b-v1.5`  | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)   |
| 7B-16k   | `python test_chat.py --model-path lmsys/vicuna-7b-v1.5-16k`  | [lmsys/vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k)   |
| 13B  | `python test_chat.py --model-path lmsys/vicuna-13b-v1.5` | [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| 13B-16k  | `python test_chat.py --model-path lmsys/vicuna-13b-v1.5-16k` | [lmsys/vicuna-13b-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k) |
| 33B  | `python test_chat.py --model-path lmsys/vicuna-33b-v1.3` | [lmsys/vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3) |

### LLAMA
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| Llama-3.2-3B   | `python test_chat.py --model-path meta-llama/Llama-3.2-3B --model-id Llama-3.2-3B`  | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)   |


### Deepseek
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| DeepSeek-Coder-V2-Instruct   | `python test_chat.py --model-path deepseek-ai/DeepSeek-Coder-V2-Instruct --model-id DeepSeek-Coder-V2-Instruct`  | [deepseek-ai/DeepSeek-Coder-V2-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)   |
### Glm
**NOTE: `transformers==4.40.2` is required for chatglm2-6b.**
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| chatglm2-6b   | `python test_chat.py --model-path THUDM/chatglm2-6b --model-id chatglm2-6b`  | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)   |
### Openchat
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| openchat_3.5   | `python test_chat.py --model-path openchat/openchat_3.5 --model-id openchat_3.5`  | [openchat/openchat_3.5](https://huggingface.co/openchat/openchat_3.5)   |
### Mistral
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| Mistral-7B-v0.1   | `python test_chat.py --model-path mistralai/Mistral-7B-v0.1 --model-id Mistral-7B-v0.1`  | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)   |
### Qwen
| Model | Chat Command | Hugging Face Repo |
| ---  | --- | --- |
| Qwen2-72B   | `python test_chat.py --model-path Qwen/Qwen2-72B --model-id Qwen2-72B`  | [Qwen/Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B)   |


# How to download model on huggingFace
## Huggingface
###  Install
```bash
pip install huggingface_hub
```

### Export Mirror
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Login in
```bash
huggingface-cli login --token "..."
```
### 4090 Download
huggingface-cli download --resume-download openchat/openchat-3.5-0106 --local-dir /home/data/model/openchat-3.5-0106

### Download Model(LLAMA3)
```bash
huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct --local-dir /home/model/meta-llama/Llama-3.1-8B-Instruct
```
#### lucadiliello/BLEURT-20
huggingface-cli download --resume-download lucadiliello/BLEURT-20 --local-dir /home/model/BLEURT-20
#### tencent/Hunyuan-7B-Instruct
huggingface-cli download --resume-download tencent/Hunyuan-7B-Instruct --local-dir /home/model/Hunyuan-7B-Instruct
#### mistralai/Mistral-Small-24B-Instruct-2501
huggingface-cli download --resume-download mistralai/Mistral-Small-24B-Instruct-2501 --local-dir /home/model/Mistral-Small-24B-Instruct-2501
#### facebook/opt-1.3b
huggingface-cli download --resume-download facebook/opt-1.3b --local-dir /home/model/opt-1.3b
#### openai-community/gpt2
huggingface-cli download --resume-download openai-community/gpt2 --local-dir /home/model/gpt2
#### Skywork/Skywork-Critic-Llama-3.1-8B
huggingface-cli download --resume-download Skywork/Skywork-Critic-Llama-3.1-8B --local-dir /home/model/Skywork-Critic-Llama-3.1-8B
#### BAAI/JudgeLM-13B-v1.0
huggingface-cli download --resume-download BAAI/JudgeLM-13B-v1.0 --local-dir /home/model/JudgeLM-13B-v1.0
#### Qwen/Qwen2.5-14B-Instruct
huggingface-cli download --resume-download Qwen/Qwen2.5-14B-Instruct --local-dir /home/model/Qwen2.5-14B-Instruct
#### prometheus-eval/prometheus-7b-v2.0
huggingface-cli download --resume-download prometheus-eval/prometheus-7b-v2.0 --local-dir /home/model/prometheus-7b-v2.0
#### WeOpenML/PandaLM-7B-v1
huggingface-cli download --resume-download WeOpenML/PandaLM-7B-v1 --local-dir /home/model/PandaLM-7B-v1
#### GAIR/autoj-13b
huggingface-cli download --resume-download GAIR/autoj-13b --local-dir /home/model/autoj-13b
#### mistralai/Mistral-7B-Instruct-v0.2
huggingface-cli download --resume-download mistralai/Mistral-7B-Instruct-v0.2 --local-dir /home/model/Mistral-7B-Instruct-v0.2

### Download Dataset(COCO)
EdinburghNLP/xsum
```bash
huggingface-cli download --repo-type dataset --resume-download WeixiangYan/CodeTransOcean --local-dir /home/dataset/val2014 --local-dir-use-symlinks False
```

#### Download JudgeBench
```bash
huggingface-cli download --repo-type dataset --resume-download ScalerLab/JudgeBench --local-dir /home/dataset/JudgeBench --local-dir-use-symlinks False
```

huggingface-cli download --repo-type dataset --resume-download EdinburghNLP/xsum --local-dir /home/model/dataset/xsum --local-dir-use-symlinks False
##### iohadrubin/wikitext-103-raw-v1
huggingface-cli download --repo-type dataset --resume-download iohadrubin/wikitext-103-raw-v1 --local-dir /home/model/dataset/wikitext-103-raw-v1 --local-dir-use-symlinks False


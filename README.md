# RobustJudge
This repository provides the official codebase for the paper "LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge."

## About
### 📚 Resources
- **[Paper](https://arxiv.org/pdf/2506.09443):** Details the framework's design and key experimental results.

## 🛠️ Setup
```bash
conda create --name robustjudge python=3.10
conda activate robustjudge
pip install -r requirements.txt
export OPENAI_API_KEY="..."

```

## 💻 Usage examples
1. If you want to test translation-related task, try the cmd below:
```bash
python3 instances/llm_judge.py --dataset flores200 --data-dir /dataset/flores200/dev --attack autodan --judge score --target-model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --target-model-id "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --output-file paper_results/temp --source "Chinese (Simplified)" --target English --num-items 1
```
2. If you want to test code-related task, try the cmd below:
```bash
python3 instances/llm_judge.py --data-dir dataset/code_task/code2code/multilingual_train.json --attack tap --task code_translation --output-file result/temp --dataset code2code --judge score
```
3. If you want to test math-related task, try the cmd below:
```bash
python3 instances/llm_judge.py --data-dir dataset/math_task/livebench-math.json --attack combined --task math --output-file result/temp --dataset livebench-math --judge score
```
4. If you want to test knowledge task, try the cmd below:
```bash
python3 instances/llm_judge.py --data-dir dataset/knowledge_task/mmlu-knowledge.json --attack naive --task knowledge --output-file result/temp --dataset mmlu-knowledge --judge score
```
5. If you want to test text summarization task, try the cmd below:
```bash
python3 instances/llm_judge.py --data-dir dataset/summarization_task/cnn_dailymail.json --task summarization --attack cheating_llm_attack --output-file paper_results/temp --dataset cnn_dailymail --judge score --target-model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --target-model-id "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" --num-items 1 
```

## 🖊️ Citing RobustJudge
```bibtex
@misc{li2025llmsreliablyjudgeyet,
      title={LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge}, 
      author={Songze Li and Chuokun Xu and Jiaying Wang and Xueluan Gong and Chen Chen and Jirui Zhang and Jun Wang and Kwok-Yan Lam and Shouling Ji},
      year={2025},
      eprint={2506.09443},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.09443}, 
}

```
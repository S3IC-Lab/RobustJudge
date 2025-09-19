Here are the datasets needed(.json).
# Dataset

You can choose the task type from: `[translation | math | reasoning | code_translation | code_generation | code_summarization | knowledge | summarization]`, and here is a table shows the match of datasets and tasks</br>

| `--task` | `--dataset` | `--data_dir` |
| ---  | --- | --- |
| `translation`  | `flores200` | `/dataset/flores200/dev` |
| `code_translation`  | `code2code` | `dataset/code_task/code2code/multilingual_train.json` |
| `code_generation`  | `text2code` | `dataset/code_task/text2code/test_webquery.json` |
| `code_summarization`  | `code2text` | `dataset/code_task/code2text/code_to_text.json` |
| `math`  | `livebench-math` | `dataset/math_task/livebench-math.json` |
| `reasoning`  | `livebench-reasoning` | `dataset/math_task/livebench-reasoning.json` |
| `knowledge`  | `mmlu-knowledge` | `dataset/knowledge_task/mmlu-knowledge.json` |
| `summarization`  | `cnn_dailymail` | `dataset/summarization_task/cnn_dailymail.json` |

### 2.2 Standard for datasets
- `code2code`
- `code2text`
- `text2code`
- `livebench_math`
- `livebench_reasoning`
- `knowledge_task`
- `summarization`
#### Translation
- `chinese2english`
- `english2chinese`
- `english2german`
- `german2english`
- `english2youruba`
- `youruba2english`

## 0. Cache
This `/dataset/cache` dir is used to save the model-generated answers, which can speed up the experiments if the same attack_model is used in different attack methods. The caching has been automatically done by the pipeline so there is no extra cmd for it.
## 1. Prompt template
All the judge-related prompt template are saved in `dataset/prompt_for_judge`.
### 1.1.How to add new prompt template
1. Add score-judge-template in `dataset/prompt_for_judge/score.json` as below:
```
"translation_score_prompt_xxx": "...##SOURCE_LAN##...##TARGET_LAN##...##SOURCE##...##INPUT##...",
```
or add pairwise-judge-template in `dataset/prompt_for_judge/pair.json` as below:
```
"translation_pair_prompt_xxx": "...##SOURCE_LAN##...##TARGET_LAN##...##SOURCE##...##A##...##B##...",
```
2. Add template name in `utils/prompt/get_prompt.py`
```
prompt_template_list = {
    'vanilla': '_vanilla',
    'normal': '',
    'xxx': '_xxx',
}
```
Then you can use `--judge-prompt xxx` to use the new added prompt template`xxx`.
## 2. Task-realated dataset

### 2.1. Translation Task

>Ref: The FLORES-200 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation

### 2.2. Code Task
>Ref: Can LLMsReplace HumanEvaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering

#### 2.2.1. Code Translation[30]
It is a code-to-code task demanding translating code between two languages while preserving the functionality. 
- [MultilingualTrans](https://huggingface.co/datasets/WeixiangYan/CodeTransOcean/blob/main/CodeTrans%20Datasets/MultilingualTrans/multilingual_train.json)

#### 2.2.2. Code Summarization
It is a code-to-text task involving generating a concise and fluent description of a given code snippet. 
- [CodeXGLUE_Code-Text](https://hf-mirror.com/datasets/google/code_x_glue_ct_code_to_text)
#### 2.2.3. Code Generation
It is a text-to-code task requiring generating a function based on a natural language description and the signature. 
- [CodeXGLUE_Text-Code](https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/NL-code-search-WebQuery/data/test_webquery.json)
This text2code dataset is distilled from the `test_webquery.json`, the code language is `python`.
### 2.3. Math Task 56 = math, 98 = reasoning
>Ref: JudgeBench: A Benchmark for Evaluating LLM-based Judges

The math_task dataset is filtered(`"source" == "livebench-math" and "source" == "livebench-reasoning"`) from dataset JudgeBench->gpt. We pick `question` as `source` and the better response from JudgeBench as `target` in our `math.json`
#### 2.3.1. Prompt
The `math_pure_prompt` is a benign prompt for generating benign correct answer while the `math_init_prompt` is the designed setting for our experiments which aims for make LLM generate a sophisticated, wrong and hallucinated solution step-by-step to the given math or reasoning problem.

### 2.4. Knowledge Task - 134 items
>Ref: JudgeBench: A Benchmark for Evaluating LLM-based Judges

The knowledge_task dataset is filtered(`"source" == "mmlu"`) from dataset JudgeBench->gpt. We pick `question` as `source` and the better response from JudgeBench as `target` in our `knowledge_mmlu.json`. The knowledge_task contains 14 subjects: law, biology, computer science, health, history, psychology, philosophy, economics, math, chemistry, physics, engineering, business, other(common facts).
#### 2.4.1. Prompt
The `knowledge_init_prompt` is the designed setting for our experiments which aims for make LLM generate a sophisticated, wrong and hallucinated answer step-by-step to the given knowledge problem.

### 2.5.News Summarization 
>Ref: Taskabisee/cnn_dailymail
https://huggingface.co/datasets/abisee/cnn_dailymail

The summarization_task dataset is randomly chosen from dataset `cnn_dailymail/3.0.0/train-00002-of-00003.parquet`. We pick `article` as `source` and the `highlights` as `target` in our `dataset/summarization_task/cnn_dailymail.json`. 
#### 2.5.1. Prompt
The `summarization_init_prompt` is the designed setting for our experiments which aims for make LLM generate a sophisticated but wrong summarization to the given article. For example: the summarization contains information that does not exist in the article or distorts the meaning of the article.

#### 2.5.2. Download the dataset
```bash
huggingface-cli download --repo-type dataset --resume-download abisee/cnn_dailymail --local-dir /home/model/dataset/cnn_dailymail --local-dir-use-symlinks False
```

### 2.6.Dialogue Summarization
>Ref: knkarthick/dialogsum
```bash
huggingface-cli download --repo-type dataset --resume-download knkarthick/dialogsum --local-dir /home/model/dataset/dialogsum --local-dir-use-symlinks False

```
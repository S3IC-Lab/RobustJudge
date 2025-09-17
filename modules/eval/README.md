## Evaluator
### 1. CodeBleu
>ref: Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering

1. Install Codebleu
```
git clone https://github.com/k4black/codebleu.git
```
2. Install requirements
```
pip install -e .
```
3. Install tree-sitter
```bash
pip install tree_sitter_go==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tree_sitter_python==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tree_sitter_cpp==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tree_sitter_c==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tree_sitter_c_sharp==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tree_sitter_java==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
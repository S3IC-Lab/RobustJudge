## Universal Adversarial Attack
>[1] Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment

This attack is adapted from [attack-comparative-assessment](https://github.com/rainavyas/attack-comparative-assessment).
### 0. Threat model
To impose this uni_adv_attack, we need to have: 
1. **Attack model**, as the surrogate model so that we could get loss feedback and the learned attack phrase. Then we can use this learned attack phrase to attack the **target model**.
2. ability to insert the attack phrase to the judge prompt as the malicious answer.

### 1. Use attack model to find attack phrase
Here are the required settings used in our experiments: 
- **i**. the number of the training dataset is 2(the last 2 items of the task dataset, **Note**: each task will be trained seperatedly to find their suitable attack phrase.)
- **ii**. attack model OpenChat-3.5-0106 as surrogate model 
- **iii**. the algorithm Greedy Search as given in [1](Algorithm 1), to find the universal attack phrase(best malicious suffix) that will get higher score in Judge score mode or be picked out as the better one in Judge pairwise mode. **Note**: each Judge mode will be trained seperatedly to find their suitable attack phrase.

### 2. Make malicious answer
Add attack phrase to the end of initial answer and sewed it into Judge prompt as example below:
- Judge score prompt with uni_adv_attack:
```
You are tasked with evaluateing a translation from English to German...
[Original English Text]
...
[German Translation]
{initial_answer + attack_phrase_score}
Please...
``` 
- Judge pairwise prompt with uni_adv_attack:
```
You are tasked with comparing two translations from English to German...
**Original English Text**:
...
**Translation A**:
...
**Translation B**:
{initial_answer + attack_phrase_pair}

Please...
``` 
**Note**: Here we use the cache data generated in plain attack(with no attack) as the training data to find the attack_phrase. As mentioned in [1], "powerful large models (GPT3.5) are more susceptible when attacked by shorter phrases, possibly because longer phrases may begin to overfit the properties of the surrogate model." So we choose the `num_attack_phrase_words = 3`.
### 3. Get result
Give those prompts mentioned in step 2 to target model and get score / better one.

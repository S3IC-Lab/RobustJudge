## Attack
You can choose the attack method from: `[none | naive | ignore | combined | escape | reasoning | completion | empty | adv | cheating | uni | gcg | autodan | pair | tap]`, the default option is `[none]`, which means no attack method is added into the pipeline. </br>
**NOTE**: Attack method: `adv | pair | tap` does not support task: `[math | reasoning | knowledge]`

3. You can choose the judge method from: `[score | pairwise]` to start your experiments.

4. Run the experiment using the bash.generate.sh file. I’ve included specific examples and notes in the file’s comments,following the steps to run it.
### 1. How to contribute
1. Registry
Add your attack-name in your `modules/attack/attack_registry.py`, For example:
```
AttackRegistry.register("translation", "gcg")(TranslationGCGAttack)
```
2. Add new class in your `modules/attack/new_attack/xxx.py`
Inherit base class `JudgeAttacker` or XXXPlain, and add new attack for task XXX:
```
class XXXNewAttack(XXXPlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(XXXNewAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return get_XXX_pure_prompt(self.args)

    def get_prompt_with_source(self, source):
        return get_XXX_init_prompt(source)
     
    def att_model_generate(self, prompt, item):
        ...
        
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass   
```
3. Import your new XXXNewAttack from you `modules/attack/new_attack/xxx.py` in `__init__.py`
```
from modules/attack/new_attack/xxx.py import XXXNewAttack
```
4. Add your new_attack name into the `attack_list` of the file: `modules/data/data_save.py` to match which prefix of the model_generated_cache you want to use:
```
attack_list = {
    ...
    'new_attack': 'manual'
}
```
# attack_executor.py
from modules.attack import AttackRegistry

class AttackFactory:
    @staticmethod
    def create_attack(self):
        attack_cls = AttackRegistry.get_attack(self.args.task, self.args.attack)
        return attack_cls(self.evaluator, self.attack_instance, self.target_instance, self.args)

class AttackExecutor:
    def __init__(self, benchmark_evaluator, attack_instance, target_instance, args):
        self.args = args
        self.task_type = args.task
        self.attack_type = args.attack
        self.evaluator = benchmark_evaluator
        self.attack_instance = attack_instance
        self.target_instance = target_instance
        
    def create(self):
        return AttackFactory.create_attack(self)
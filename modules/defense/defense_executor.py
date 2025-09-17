# defense_executor.py
from modules.defense import DefenseRegistry

class DefenseFactory:
    @staticmethod
    def create_defense(self):
        defense_cls = DefenseRegistry.get_defense(self.args.task, self.args.defense)
        return defense_cls(self.attack_instance, self.defense_instance, self.args)

class DefenseExecutor:
    def __init__(self, attack_instance, defense_instance, args):
        self.args = args
        self.task_type = args.task
        self.defense_type = args.defense
        self.attack_instance = attack_instance
        self.defense_instance = defense_instance
        
    def create(self):
        return DefenseFactory.create_defense(self)
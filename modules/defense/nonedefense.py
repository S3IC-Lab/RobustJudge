from modules.defense.base_defense import JudgeDefender

class NoneDefense(JudgeDefender):
    def __init__(self, attack_instance, defense_instance, args):
        super().__init__(attack_instance, defense_instance, args)
    def run(self):
        #TODO: Implement the run method for nonedefense
        pass
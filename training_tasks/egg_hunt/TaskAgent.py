from base_classes.BaseAgent import BaseAgent

class TaskAgent(BaseAgent):
    def __init__(self, color=None):
        super().__init__(color=color)  
        self.found_egg = False        

    def prepare_brain_inputs(self, **raw_inputs):
        sub_view = super().prepare_brain_inputs(**raw_inputs)
        return sub_view
    
    def reset(self):
        self.alive = True
        self.found_egg = False
    
    def move(self, **raw_inputs):
        super().move(**raw_inputs)
        if self.position == raw_inputs['egg_position']:
            self.found_egg = True
        
        
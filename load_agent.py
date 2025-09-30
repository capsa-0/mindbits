from utils.yaml_manager import load_yaml_config
import importlib

def load_agent_brain(agent_path):

    agent_config = load_yaml_config(agent_path + '/brain_config.yaml')
    brain_config = agent_config.brain_config
    brain_params = load_yaml_config(agent_path + '/brain_parameters.yaml')

    brain_class_name = brain_config['brain_class']    
    module_name = f"brain_development.brains.{brain_class_name.lower()}"  
    module = importlib.import_module(module_name)
    brain_class = getattr(module, brain_class_name)
    brain = brain_class(brain_config)
    brain.load(brain_params)

    return brain

def create_agent_brain(brain_config):

    brain_class_name = brain_config.brain_class    
    module_name = f"brain_development.brains.{brain_class_name}"  
    module = importlib.import_module(module_name)
    brain_class = getattr(module, brain_class_name)
    brain = brain_class(brain_config)
    brain.initialize_network()

    return brain

def load_agent(agent_path, brain, task):
    agent_config = load_yaml_config(agent_path + '/agent_config.yaml')

    module_name = f"training_tasks.{task}.TaskAgent"
    module = importlib.import_module(module_name)
    agent_class = getattr(module, "TaskAgent")

    agent = agent_class(brain=brain, agent_config=agent_config)

    return agent

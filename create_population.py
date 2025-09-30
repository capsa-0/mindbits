import os
import time
import yaml

from utils.yaml_manager import load_yaml_config
from load_agent import create_agent_brain
from utils.functions import get_folder_size

from base_classes.BaseAgent import BaseAgent

def create_initial_population(initial_population_yaml):

    initial_population_config = load_yaml_config(initial_population_yaml)

    population_path = 'populations' + os.sep + initial_population_config.name
    os.makedirs(population_path, exist_ok=True)

    config_file_path = population_path + os.sep + 'agents_config.yaml'
    open(config_file_path, "w").close()

    brain_params_path = population_path + os.sep + 'brain_parameters'
    os.makedirs(brain_params_path, exist_ok=True)

    j = 0


    brain_configs = []
    for batch_name, batch_cfg in vars(initial_population_config.batchs).items():
        brain_configs.append(batch_cfg.agent_config.brain_config)
        for i in range(batch_cfg.amount):

            agent_config = batch_cfg.agent_config  
            agent_config.batch = batch_name
            j += 1    
            agent_config.id = j
            brain = create_agent_brain(agent_config.brain_config)
            agent = BaseAgent(brain=brain, agent_config=agent_config)

            agent.save_agent_config(config_file_path)
            brain.save(brain_params_path + os.sep + f'brain_params_{j}.pt')


    metadata_file_path = population_path + os.sep + 'metadata.yaml'
    os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)

    creation_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    brain_configs_set = {str(bc) for bc in brain_configs}
    brain_types = len(brain_configs_set)
    topology_purity = "homogeneous" if brain_types == 1 else "mixed"
    population_size_mb = round((get_folder_size(population_path) / (1024 * 1024)), 3)

    metadata_content = {
    "population_name": initial_population_config.name,
    "population_size": initial_population_config.total_size,
    "creation_date": creation_date ,
    "description": initial_population_config.description,
    "brain_purity": topology_purity,
    "author": initial_population_config.author,
    "size_mb": population_size_mb,
    "recent_tasks": []
    }

    with open(metadata_file_path, 'w') as file:
        yaml.dump(metadata_content, file, sort_keys=False)

 
folder_path = 'initial_population_configs'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):  
        create_initial_population(file_path)







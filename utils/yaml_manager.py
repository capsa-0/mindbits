import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

# Cargar YAML como objeto
def load_yaml_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return dict_to_namespace(data)

def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(namespace_to_dict(v) for v in obj)
    else:
        return obj
    
def dump_config(config, filepath):
    dict_config = namespace_to_dict(config)
    with open(filepath, "w") as f:
        yaml.safe_dump(dict_config, f, sort_keys=False, default_flow_style=True)

def dump_agents_config(agents, filepath):

    with open(filepath, "w") as f:
        for agent in agents:
            # convertir a dict si es SimpleNamespace
            dict_agent = namespace_to_dict(agent)
            yaml.safe_dump(dict_agent, f, sort_keys=False, default_flow_style=True)

def dump_line(agent_config, filepath):
    """
    Agrega un agente (SimpleNamespace o dict) como línea compacta al YAML.
    """
    dict_agent = namespace_to_dict(agent_config)
    with open(filepath, "a") as f:
        yaml.safe_dump(dict_agent, f, sort_keys=False, default_flow_style=True)
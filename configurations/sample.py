from typing import Dict, Any
from syne_tune.config_space import loguniform, uniform, choice


def configuration_space() -> Dict[str, Any]:
    return {
        "learning_rate": loguniform(1e-4, 1e-1),
        "weight_decay": loguniform(1e-5, 1e-1),
        "swa_replace_frequency": choice([-1, 10, 50, 100, 200, 400]),
        "swa_gamma": uniform(0.1, 1.0),
        "hidden_dim": choice([2, 24, 48, 92]),
        "num_layers": choice([1, 2]),
        "gpu": 0, # Set to 1 to use GPU
        "plot": 0,
        
    }

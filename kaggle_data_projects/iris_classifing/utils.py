import yaml
from typing import Dict


def load_config_file(file_name: str) -> Dict:

    with open(file_name, "r") as file:
        config_content = yaml.safe_load(file)

    return config_content

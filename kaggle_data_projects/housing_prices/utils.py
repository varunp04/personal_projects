import os
import pickle
import yaml
from typing import Dict, List


def load_config_file(file_location: str) -> Dict:

    with open(file_location, "r") as file:
        config_file_content = yaml.safe_load(file)

    return config_file_content


def save_as_pickle(path: str, artifact_name: str, artifact):

    with open(os.path.join(path, artifact_name), "wb") as out:
        pickle.dump(artifact, out)


def save_as_yaml(path: str, artifact_name: str, artifact):

    with open(os.path.join(path, artifact_name), "w") as out:

        yaml.dump(artifact, out)

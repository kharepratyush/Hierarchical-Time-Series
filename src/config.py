"""
Author: Pratyush Khare
"""

import yaml
from .custom_exception import HTSCustomException


def configuration_setup() -> object:
    """

    @rtype: object
    """
    with open("config/config.yaml", "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise HTSCustomException("Issue with YAML set-up. Please resolve.")

        return configuration

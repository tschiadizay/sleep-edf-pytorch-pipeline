"""Contain the logging infrastructure used in the entire project."""
import logging
import logging.config
from pathlib import Path

import yaml


def set_up_logging(
        logging_config_path : str | Path = "utils/logging_config.yaml"):
    """Load logging config and initialize root logger.

    Args:
        logging_config_path (str | Path): Path to the logging_config.yaml file.

    """
    with open(logging_config_path) as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    logging.captureWarnings(True)

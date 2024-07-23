"""
Author: Pratyush Khare
"""

import logging
import os
from typing import List


class ColorHandler(logging.StreamHandler):
    """
    Custom logging handler to output colored logs to the console.
    """

    GRAY8 = "38;5;8"
    GRAY7 = "38;5;7"
    ORANGE = "33"
    RED = "31"
    WHITE = "0"

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record with a color-coded log level.

        Parameters:
        record (logging.LogRecord): The log record to be emitted.
        """
        level_color_map = {
            logging.DEBUG: self.GRAY8,
            logging.INFO: self.GRAY7,
            logging.WARNING: self.ORANGE,
            logging.ERROR: self.RED,
        }

        csi = f"{chr(27)}["  # control sequence introducer
        color = level_color_map.get(record.levelno, self.WHITE)

        print(
            f"{csi}{color}m{record.asctime}s - {record.levelname}s - {record.msg}{csi}m"
        )


def create_folders(directories: List[str] = None) -> None:
    """
    Create required directories if they do not exist.

    Parameters:
    directories (List[str], optional): List of directories to create. Defaults to a predefined set.

    Returns:
    None
    """
    if directories is None:
        directories = [
            "data/raw",
            "data/processed",
            "models",
            "notebooks",
            "tests",
            "logs",
            "data/output",
        ]

    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Directory {directory} created.")
            else:
                logging.info(f"Directory {directory} already exists.")
        except OSError as e:
            logging.error(f"Failed to create directory {directory}: {e}")


def setup_logging(log_file: str = "logs/pipeline.log") -> None:
    """
    Set up logging configuration.

    Parameters:
    log_file (str, optional): Path to the log file. Defaults to 'logs/pipeline.log'.

    Returns:
    None
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = ColorHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger("").addHandler(console)
        logging.info("Logging is set up.")
    except OSError as e:
        logging.error(f"Failed to set up logging: {e}")

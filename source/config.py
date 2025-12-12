import argparse
import logging
import os
import sys

import yaml
from pydantic import ValidationError

from source.arguments_schema import PipelineParams

logger: logging.Logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Handles argument parsing and loading the pipeline parameters file.
    """

    def __init__(self) -> None:
        """
        Initializes the `ArgumentParser` and defines expected arguments.
        """
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="Machine Learning Pipeline for Diabetes Prediction."
        )

        self.parser.add_argument(
            "--params",
            type=str,
            default="config/params.yaml",
            help="Path to the parameters file (.yaml).",
        )

    def _load_params(self, params_path: str) -> PipelineParams:
        """
        Private helper to load the params.yaml file and validate it.

        :param params_path: Path to params.
        :return: Dictionary of pipeline params.
        """
        if not os.path.exists(params_path):
            error_msg = f"Parameters file not found at: {params_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(params_path, "r") as f:
                params_dict = yaml.safe_load(f)
                if params_dict is None:
                    params_dict = {}

            params_obj: PipelineParams = PipelineParams(**params_dict)

            logger.info(f"Successfully loaded params from {params_path}")
            return params_obj

        except ValidationError as e:
            logger.error("--- Invalid 'params.yaml' configuration! ---")
            logger.error(f"File: {params_path}")
            logger.error(f"{e}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Failed to parse params file: {e}", exc_info=True)
            raise e

    def parse_config(self) -> tuple[argparse.Namespace, PipelineParams]:
        """
        Public method to parse arguments and load the parameters file.

        :return: A tuple of command line arguments and loaded params.
        """
        args: argparse.Namespace = self.parser.parse_args()
        params: PipelineParams = self._load_params(args.params)

        return args, params

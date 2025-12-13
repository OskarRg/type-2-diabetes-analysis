import logging

from source.config import ConfigParser
from source.logging_config import setup_logging
from source.pipeline import Pipeline

setup_logging(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config_parser = ConfigParser()
    args, params = config_parser.parse_config()

    pipeline: Pipeline = Pipeline(params)

    pipeline.run(data_path=params.data.input_path)

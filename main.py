import logging

from source.logging_config import setup_logging
from source.pipeline import Pipeline

setup_logging(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # TODO Create config and CLI argument parser
    pipeline = Pipeline()
    pipeline.run(data_path="data/diabetes.csv")

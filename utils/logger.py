import logging
import os
from config import LOG_DIR


# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)


logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def log(msg):
    print(msg)
    logging.info(msg)
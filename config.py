import os

MODEL = "gpt-4o"
EMBED_MODEL = "all-MiniLM-L6-v2"

MAX_ITER = 5
TARGET_SCORE = 4.5

OUTPUT_DIR = "maf_run"

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
CODE_DIR = os.path.join(OUTPUT_DIR, "codes")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

DIRS = [OUTPUT_DIR, LOG_DIR, IMG_DIR, CODE_DIR, DATA_DIR]
from bootstrap import *
import os
import json
from config import DATA_DIR, CODE_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

def save_json(name, obj):

    path = os.path.join(DATA_DIR, name + ".json")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_text(name, text):

    path = os.path.join(DATA_DIR, name + ".txt")

    with open(path, "w") as f:
        f.write(text)


def save_code(name, code):

    path = os.path.join(CODE_DIR, name + ".py")

    with open(path, "w") as f:
        f.write(code)
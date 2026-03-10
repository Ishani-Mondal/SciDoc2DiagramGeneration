from bootstrap import *
import subprocess
import shutil
import os
from config import CODE_DIR, IMG_DIR
from utils.logger import log

def render_diagram(code, name):

    path = os.path.join(CODE_DIR, name + ".py")

    with open(path, "w") as f:
        f.write(code)

    subprocess.run(["python", path], check=True)

    img_src = "diagram.png"
    img_dst = os.path.join(IMG_DIR, name + ".png")

    shutil.copy(img_src, img_dst)

    log(f"Saved image {img_dst}")

    return img_dst
from bootstrap import *
from google import genai
from PIL import Image
from utils.io_utils import save_text

gemini_client = genai.Client(api_key="AIzaSyCD-jC2tSyL8TZUbyf5cyYiWIqtDaxm1kQ")

def call_gemini_image(prompt, image_path, name=None):

    img = Image.open(image_path)

    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )

    out = resp.text

    if name:
        save_text(name + "_vision_prompt", prompt)
        save_text(name + "_vision_response", out)

    return out